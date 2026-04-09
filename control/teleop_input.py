from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from control.teleop_config import TeleopConfig


@dataclass
class GamepadSample:
    """
    单次轮询得到的手柄状态快照。
    """

    timestamp_sec: float
    axis_x: float
    axis_y: float
    forward_pressed: bool
    estop_pressed: bool
    reset_primary_pressed: bool
    reset_secondary_pressed: bool
    connected: bool


class PygameGamepadInput:
    """
    pygame controller 输入后端。

    异常行为:
    - 初始化失败/未发现手柄会抛 RuntimeError。
    - 轮询失败时返回 connected=False 的样本，交由上层安全逻辑处理。
    """

    _BUTTON_ATTR_BY_NAME = {
        "south": "CONTROLLER_BUTTON_A",
        "east": "CONTROLLER_BUTTON_B",
        "west": "CONTROLLER_BUTTON_X",
        "north": "CONTROLLER_BUTTON_Y",
        "share": "CONTROLLER_BUTTON_BACK",
        "back": "CONTROLLER_BUTTON_BACK",
        "options": "CONTROLLER_BUTTON_START",
        "start": "CONTROLLER_BUTTON_START",
        "guide": "CONTROLLER_BUTTON_GUIDE",
        "leftstick": "CONTROLLER_BUTTON_LEFTSTICK",
        "rightstick": "CONTROLLER_BUTTON_RIGHTSTICK",
        "leftshoulder": "CONTROLLER_BUTTON_LEFTSHOULDER",
        "rightshoulder": "CONTROLLER_BUTTON_RIGHTSHOULDER",
        "dpad_up": "CONTROLLER_BUTTON_DPAD_UP",
        "dpad_down": "CONTROLLER_BUTTON_DPAD_DOWN",
        "dpad_left": "CONTROLLER_BUTTON_DPAD_LEFT",
        "dpad_right": "CONTROLLER_BUTTON_DPAD_RIGHT",
    }
    # SDL2 GameController 轴值通常来自 Sint16: [-32768, 32767]
    _SDL_AXIS_FULL_SCALE = 32768.0

    def __init__(
        self,
        cfg: TeleopConfig,
    ) -> None:
        self.cfg = cfg
        self.pygame = None
        self.controller_mod = None
        self.controller = None
        self.controller_index: Optional[int] = None
        self.controller_name: str = ""

        self._button_ids: Dict[str, int] = {}

    def _import_backend(self) -> None:
        if self.pygame is not None and self.controller_mod is not None:
            return

        try:
            import pygame
            from pygame._sdl2 import controller as sdl2_controller
        except ImportError as exc:
            raise RuntimeError(
                "pygame controller backend is unavailable. Install with: pip install pygame>=2.5"
            ) from exc

        self.pygame = pygame
        self.controller_mod = sdl2_controller

    def initialize(self) -> None:
        self._import_backend()
        assert self.pygame is not None
        assert self.controller_mod is not None

        self.pygame.init()
        self.pygame.joystick.init()
        self.controller_mod.init()

    def list_controllers(self) -> Sequence[Tuple[int, str, bool]]:
        self.initialize()
        assert self.controller_mod is not None

        out = []
        count = int(self.controller_mod.get_count())
        for idx in range(count):
            is_controller = bool(self.controller_mod.is_controller(idx))
            name = "UNKNOWN"
            if is_controller:
                temp = None
                try:
                    temp = self.controller_mod.Controller(idx)
                    name = str(getattr(temp, "name", f"controller-{idx}"))
                except Exception:
                    name = f"controller-{idx}"
                finally:
                    if temp is not None:
                        self._safe_close_controller(temp)
            out.append((idx, name, is_controller))
        return out

    def open(self) -> None:
        self.initialize()
        assert self.controller_mod is not None

        candidates = []
        for idx, name, is_controller in self.list_controllers():
            if is_controller:
                candidates.append((idx, name))

        if len(candidates) == 0:
            raise RuntimeError("No SDL game controller found.")

        desired_idx = int(self.cfg.controller_index)
        selected_index: Optional[int] = None
        selected_name: Optional[str] = None

        if self.cfg.controller_name_contains:
            kw = self.cfg.controller_name_contains.lower()
            for idx, name in candidates:
                if kw in name.lower():
                    selected_index = idx
                    selected_name = name
                    break

        if selected_index is None:
            for idx, name in candidates:
                if idx == desired_idx:
                    selected_index = idx
                    selected_name = name
                    break

        if selected_index is None:
            selected_index, selected_name = candidates[0]

        self.controller = self.controller_mod.Controller(selected_index)
        self.controller_index = int(selected_index)
        self.controller_name = str(selected_name)

        self._button_ids = {
            "forward": self._resolve_button_id(self.cfg.forward_button),
            "estop": self._resolve_button_id(self.cfg.estop_button),
            "reset_primary": self._resolve_button_id(self.cfg.reset_combo[0]),
            "reset_secondary": self._resolve_button_id(self.cfg.reset_combo[1]),
        }

    def _resolve_button_id(self, button_name: str) -> int:
        assert self.pygame is not None
        name = str(button_name).strip().lower()
        attr = self._BUTTON_ATTR_BY_NAME.get(name)
        if attr is None:
            raise ValueError(
                f"Unsupported button semantic '{button_name}'. "
                f"Supported: {sorted(self._BUTTON_ATTR_BY_NAME.keys())}"
            )
        if not hasattr(self.pygame, attr):
            raise ValueError(f"pygame does not expose button constant `{attr}`.")
        return int(getattr(self.pygame, attr))

    def poll(self) -> GamepadSample:
        assert self.pygame is not None
        if self.controller is None:
            raise RuntimeError("Controller is not opened.")

        now_sec = time.monotonic()
        try:
            # pump 事件队列，让 get_axis/get_button 读到最新状态
            self.pygame.event.pump()

            connected = self._is_controller_connected(self.controller)
            if not connected:
                return GamepadSample(
                    timestamp_sec=now_sec,
                    axis_x=0.0,
                    axis_y=0.0,
                    forward_pressed=False,
                    estop_pressed=False,
                    reset_primary_pressed=False,
                    reset_secondary_pressed=False,
                    connected=False,
                )

            raw_axis_x = self.controller.get_axis(self.cfg.left_stick_x_axis)
            raw_axis_y = self.controller.get_axis(self.cfg.left_stick_y_axis)
            axis_x = self._normalize_axis_value(raw_axis_x)
            axis_y = self._normalize_axis_value(raw_axis_y)
            forward_pressed = bool(self.controller.get_button(self._button_ids["forward"]))
            estop_pressed = bool(self.controller.get_button(self._button_ids["estop"]))
            reset_primary_pressed = bool(
                self.controller.get_button(self._button_ids["reset_primary"])
            )
            reset_secondary_pressed = bool(
                self.controller.get_button(self._button_ids["reset_secondary"])
            )
        except Exception:
            # 读取异常按断连处理，由上层触发安全动作
            return GamepadSample(
                timestamp_sec=now_sec,
                axis_x=0.0,
                axis_y=0.0,
                forward_pressed=False,
                estop_pressed=False,
                reset_primary_pressed=False,
                reset_secondary_pressed=False,
                connected=False,
            )

        return GamepadSample(
            timestamp_sec=now_sec,
            axis_x=axis_x,
            axis_y=axis_y,
            forward_pressed=forward_pressed,
            estop_pressed=estop_pressed,
            reset_primary_pressed=reset_primary_pressed,
            reset_secondary_pressed=reset_secondary_pressed,
            connected=True,
        )

    @classmethod
    def _normalize_axis_value(cls, raw_axis: Any) -> float:
        """
        归一化轴值到 [-1, 1]。

        兼容两类输入:
        - 已归一化浮点: [-1, 1]
        - SDL2 原始整型: 约 [-32768, 32767]
        """
        try:
            raw = float(raw_axis)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(raw):
            return 0.0

        if -1.0 <= raw <= 1.0:
            return raw

        normalized = raw / cls._SDL_AXIS_FULL_SCALE
        return max(-1.0, min(1.0, float(normalized)))

    @staticmethod
    def _is_controller_connected(controller_obj: Any) -> bool:
        if hasattr(controller_obj, "get_attached"):
            try:
                return bool(controller_obj.get_attached())
            except Exception:
                return False
        if hasattr(controller_obj, "attached"):
            try:
                attached_attr = getattr(controller_obj, "attached")
                if callable(attached_attr):
                    return bool(attached_attr())
                return bool(attached_attr)
            except Exception:
                return False
        return True

    def close(self) -> None:
        if self.controller is not None:
            self._safe_close_controller(self.controller)
            self.controller = None
        if self.controller_mod is not None:
            try:
                self.controller_mod.quit()
            except Exception:
                pass
        if self.pygame is not None:
            try:
                self.pygame.quit()
            except Exception:
                pass

    @staticmethod
    def _safe_close_controller(controller_obj: Any) -> None:
        for attr in ("quit", "close"):
            if hasattr(controller_obj, attr):
                try:
                    fn = getattr(controller_obj, attr)
                    if callable(fn):
                        fn()
                except Exception:
                    pass
