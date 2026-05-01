"""High-level BookArm model.

This module owns high-level robot intent, Pinocchio-based kinematics, joint
limits, and gripper intent. It does not create serial ports, build low-level
ESP32 transports, or know firmware transport details.

Hardware execution is delegated to actuator objects:

    BookArm -> ArmActuator / GripperActuator -> ESP32 JSON transport
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Literal, TYPE_CHECKING

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - depends on local conda env
    raise ImportError(
        "BookArm kinematics requires pinocchio. Activate the bookarm-beiyu "
        "environment or install pinocchio from conda-forge."
    ) from exc

if TYPE_CHECKING:
    from bookarm_control_py.actuator.arm import ArmActuator
    from bookarm_control_py.actuator.gripper import GripperActuator


DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "bookarm_urdf"
    / "urdf"
    / "bookarm_urdf.urdf"
)

GripperAction = Literal["open", "close", "angle", "feedback", "torque"]


@dataclass(frozen=True)
class IKResult:
    """Inverse kinematics result."""

    success: bool
    q: np.ndarray
    error_norm: float
    iterations: int


@dataclass(frozen=True)
class GripperCommand:
    """High-level gripper intent, without ESP32 command ids."""

    action: GripperAction
    value: float | None = None


class Gripper:
    """High-level gripper intent builder."""

    def open(self) -> GripperCommand:
        return GripperCommand(action="open")

    def close(self) -> GripperCommand:
        return GripperCommand(action="close")

    def set_angle(self, angle_deg: float) -> GripperCommand:
        return GripperCommand(action="angle", value=float(angle_deg))

    def feedback(self) -> GripperCommand:
        return GripperCommand(action="feedback")

    def set_torque(self, torque: float) -> GripperCommand:
        return GripperCommand(action="torque", value=float(torque))


class BookArm:
    """High-level BookArm robot model based on Pinocchio.

    Parameters
    ----------
    urdf_path:
        URDF path. Defaults to ``assets/bookarm_urdf/urdf/bookarm_urdf.urdf``.
    end_effector_link:
        End-effector link used by FK/IK. In this project, ``link5`` is the real
        end-effector pose.
    arm_actuator:
        Optional low-level arm actuator. BookArm only calls its public methods.
    gripper_actuator:
        Optional low-level gripper actuator. BookArm only calls its public
        methods.
    gripper:
        Optional high-level gripper intent builder.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        end_effector_link: str = "link5",
        arm_actuator: "ArmActuator | None" = None,
        gripper_actuator: "GripperActuator | None" = None,
        gripper: Gripper | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF_PATH
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file does not exist: {self.urdf_path}")

        self.model = self._build_model_from_urdf(self.urdf_path)
        self.data = self.model.createData()
        self.end_effector_link = end_effector_link
        self.end_effector_frame_id = self._get_frame_id(end_effector_link)

        self.joint_names = self._get_actuated_joint_names()
        self.neutral_q = self.check_joint_angles(
            pin.neutral(self.model),
            context="Pinocchio neutral configuration",
        )
        self.gripper = gripper if gripper is not None else Gripper()
        self.arm_actuator = arm_actuator
        self.gripper_actuator = gripper_actuator

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    def connect_actuators(
        self,
        *,
        arm_actuator: "ArmActuator | None" = None,
        gripper_actuator: "GripperActuator | None" = None,
    ) -> "BookArm":
        """Connect actuator interfaces created outside this high-level model."""

        if arm_actuator is not None:
            self.arm_actuator = arm_actuator
        if gripper_actuator is not None:
            self.gripper_actuator = gripper_actuator
        return self

    def disconnect_actuators(self) -> None:
        """Disconnect connected actuators if they expose a close method."""

        closed: set[int] = set()
        for actuator in (self.arm_actuator, self.gripper_actuator):
            if actuator is None or id(actuator) in closed:
                continue
            close = getattr(actuator, "close", None)
            if callable(close):
                close()
            closed.add(id(actuator))

    def check_joint_angles(
        self,
        q: Iterable[float],
        tolerance: float = 1e-9,
        *,
        context: str = "Joint angle",
    ) -> np.ndarray:
        """Validate joint angles against URDF limits.

        Unlike clipping, this method never changes the user's command. If any
        joint is outside its allowed range, it raises ValueError.
        """

        q_array = self._as_configuration(q)
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        invalid = np.where((q_array < lower - tolerance) | (q_array > upper + tolerance))[0]

        if invalid.size:
            details = []
            for index in invalid:
                details.append(
                    f"{self.joint_names[index]}={q_array[index]:.6f} "
                    f"not in [{lower[index]:.6f}, {upper[index]:.6f}]"
                )
            raise ValueError(f"{context} out of range: " + "; ".join(details))

        return q_array

    def forward_kinematics(
        self,
        q: Iterable[float],
        end_effector_link: str | None = None,
    ) -> pin.SE3:
        """Compute end-effector pose from joint angles in radians."""

        q_array = self.check_joint_angles(q, context="FK configuration")
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        pin.forwardKinematics(self.model, self.data, q_array)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[frame_id].copy()

    def forward_kinematics_dict(
        self,
        q: Iterable[float],
        end_effector_link: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Return FK as position, rotation matrix, and homogeneous transform."""

        pose = self.forward_kinematics(q, end_effector_link=end_effector_link)
        transform = np.eye(4)
        transform[:3, :3] = pose.rotation
        transform[:3, 3] = pose.translation

        return {
            "position": pose.translation.copy(),
            "rotation": pose.rotation.copy(),
            "transform": transform,
        }

    def inverse_kinematics(
        self,
        target_position: Iterable[float],
        target_rotation: np.ndarray | None = None,
        q0: Iterable[float] | None = None,
        end_effector_link: str | None = None,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        damping: float = 1e-6,
        step_size: float = 0.4,
    ) -> IKResult:
        """Solve numerical IK while enforcing URDF joint limits."""

        target_translation = np.asarray(target_position, dtype=float).reshape(3)
        q = self._as_configuration(q0) if q0 is not None else self.neutral_q.copy()
        q = self.check_joint_angles(q, context="IK initial configuration")
        frame_id = self._get_frame_id(end_effector_link or self.end_effector_link)

        if target_rotation is None:
            return self._inverse_kinematics_position_only(
                target_translation=target_translation,
                q=q,
                frame_id=frame_id,
                max_iterations=max_iterations,
                tolerance=tolerance,
                damping=damping,
                step_size=step_size,
            )

        target_pose = pin.SE3(
            np.asarray(target_rotation, dtype=float).reshape(3, 3),
            target_translation,
        )
        return self._inverse_kinematics_pose(
            target_pose=target_pose,
            q=q,
            frame_id=frame_id,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            step_size=step_size,
        )

    def move_joints_rad(
        self,
        joint_angles_rad: Iterable[float],
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Forward high-level joint-angle intent to the connected arm actuator."""

        q = self.check_joint_angles(joint_angles_rad)
        return self._require_arm_actuator().move_joints_rad(
            q,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def move_joints_deg(
        self,
        joint_angles_deg: Iterable[float],
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        q_rad = np.deg2rad(np.asarray(list(joint_angles_deg), dtype=float))
        q_rad = self.check_joint_angles(q_rad)
        return self._require_arm_actuator().move_joints_deg(
            np.rad2deg(q_rad),
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def move_to_position(
        self,
        target_position: Iterable[float],
        *,
        target_rotation: np.ndarray | None = None,
        q0: Iterable[float] | None = None,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> IKResult:
        """Solve IK, then forward the solved joint angles to the arm actuator."""

        result = self.inverse_kinematics(
            target_position=target_position,
            target_rotation=target_rotation,
            q0=q0,
        )
        if not result.success:
            raise RuntimeError(f"IK failed with error norm {result.error_norm:.6f}")

        self.check_joint_angles(result.q)
        self._require_arm_actuator().move_joints_rad(
            result.q,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )
        return result

    def read_feedback(
        self,
        *,
        response_timeout: float | None = None,
        expected_t: int | None = None,
    ) -> dict[str, Any]:
        return self._require_arm_actuator().read_feedback(
            response_timeout=response_timeout,
            expected_t=expected_t,
        )

    def send_gripper_command(
        self,
        command: GripperCommand,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._require_gripper_actuator().send_command(
            command,
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def open_gripper(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.send_gripper_command(
            self.gripper.open(),
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def close_gripper(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.send_gripper_command(
            self.gripper.close(),
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def set_gripper_angle(
        self,
        angle_deg: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self.send_gripper_command(
            self.gripper.set_angle(angle_deg),
            wait_response=wait_response,
            response_timeout=response_timeout,
        )

    def _inverse_kinematics_position_only(
        self,
        target_translation: np.ndarray,
        q: np.ndarray,
        frame_id: int,
        max_iterations: int,
        tolerance: float,
        damping: float,
        step_size: float,
    ) -> IKResult:
        last_error_norm = float("inf")

        for iteration in range(1, max_iterations + 1):
            q = self.check_joint_angles(
                q,
                context=f"IK iteration {iteration} configuration",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_translation = self.data.oMf[frame_id].translation
            error = target_translation - current_translation
            last_error_norm = float(np.linalg.norm(error))
            if last_error_norm < tolerance:
                return IKResult(True, q, last_error_norm, iteration)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            velocity = self._damped_least_squares(jacobian, error, damping)
            q = self.check_joint_angles(
                pin.integrate(self.model, q, step_size * velocity),
                context=f"IK iteration {iteration} proposed configuration",
            )

        return IKResult(False, q, last_error_norm, max_iterations)

    def _inverse_kinematics_pose(
        self,
        target_pose: pin.SE3,
        q: np.ndarray,
        frame_id: int,
        max_iterations: int,
        tolerance: float,
        damping: float,
        step_size: float,
    ) -> IKResult:
        last_error_norm = float("inf")

        for iteration in range(1, max_iterations + 1):
            q = self.check_joint_angles(
                q,
                context=f"IK iteration {iteration} configuration",
            )
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[frame_id]
            frame_error = current_pose.actInv(target_pose)
            error = pin.log(frame_error).vector
            last_error_norm = float(np.linalg.norm(error))
            if last_error_norm < tolerance:
                return IKResult(True, q, last_error_norm, iteration)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jacobian = -pin.Jlog6(frame_error.inverse()) @ jacobian
            velocity = -self._damped_least_squares(jacobian, error, damping)
            q = self.check_joint_angles(
                pin.integrate(self.model, q, step_size * velocity),
                context=f"IK iteration {iteration} proposed configuration",
            )

        return IKResult(False, q, last_error_norm, max_iterations)

    def _get_frame_id(self, frame_name: str) -> int:
        if not self.model.existFrame(frame_name):
            available_frames = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"URDF does not contain frame/link {frame_name!r}. "
                f"Available frames: {available_frames}"
            )
        return self.model.getFrameId(frame_name)

    def _get_actuated_joint_names(self) -> list[str]:
        return [
            self.model.names[index]
            for index, joint in enumerate(self.model.joints)
            if index > 0 and joint.nq > 0
        ]

    def _as_configuration(self, q: Iterable[float]) -> np.ndarray:
        q_array = np.asarray(list(q), dtype=float)
        if q_array.shape != (self.nq,):
            raise ValueError(f"Expected {self.nq} joint values, got {q_array.shape[0]}")
        return q_array

    def _require_arm_actuator(self) -> "ArmActuator":
        if self.arm_actuator is None:
            raise RuntimeError("No arm actuator is connected.")
        return self.arm_actuator

    def _require_gripper_actuator(self) -> "GripperActuator":
        if self.gripper_actuator is None:
            raise RuntimeError("No gripper actuator is connected.")
        return self.gripper_actuator

    @staticmethod
    def _damped_least_squares(
        jacobian: np.ndarray,
        error: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        jj_t = jacobian @ jacobian.T
        damping_matrix = damping * np.eye(jj_t.shape[0])
        return jacobian.T @ np.linalg.solve(jj_t + damping_matrix, error)

    @staticmethod
    def _build_model_from_urdf(urdf_path: Path) -> pin.Model:
        if not str(urdf_path).isascii():
            return BookArm._build_model_from_ascii_copy(urdf_path)

        try:
            return pin.buildModelFromUrdf(str(urdf_path))
        except ValueError:
            return BookArm._build_model_from_ascii_copy(urdf_path)

    @staticmethod
    def _build_model_from_ascii_copy(urdf_path: Path) -> pin.Model:
        with tempfile.TemporaryDirectory(prefix="bookarm_urdf_") as temp_dir:
            temp_urdf_path = Path(temp_dir) / "bookarm_urdf.urdf"
            shutil.copy2(urdf_path, temp_urdf_path)
            return pin.buildModelFromUrdf(str(temp_urdf_path))
