"""Registry: 이름 기반 객체 등록·조회·빌드.

각 도메인(backbones, fusions, losses, ...) 은 자체 Registry 인스턴스를 만들고,
구현 클래스/함수는 import-time 에 데코레이터로 등록된다.
spec.type 문자열로 lookup 후 spec.kwargs 로 인스턴스화한다.

사용 예:
    BACKBONES = Registry("backbones")

    @BACKBONES.register("dmpnn")
    class DMPNNBackbone: ...

    backbone = BACKBONES.build_with_kwargs("dmpnn", depth=3, hidden=300)
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """이름 → 객체 매핑 컨테이너.

    Why:
        실험을 YAML 로 선언적으로 기술하려면 코드의 어떤 객체를 쓸지 문자열
        식별자로 지정할 수 있어야 한다. Registry 는 그 매핑을 일원화하고,
        등록되지 않은 키 사용 시 가용 키 목록을 친절하게 알려준다.

    Attributes:
        name: 디버그 메시지용 도메인 이름 (예: "backbones").
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    # ------------------------------------------------------------------ register
    def register(self, key: str | None = None, *, override: bool = False) -> Callable[[T], T]:
        """데코레이터 형태로 객체를 등록한다.

        Args:
            key: 등록 키. None 이면 ``obj.__name__`` 사용.
            override: 같은 키 재등록 허용 여부 (테스트용). 기본 False.

        Returns:
            원본 객체를 그대로 반환하는 데코레이터.
        """

        def deco(obj: T) -> T:
            k = key or getattr(obj, "__name__", None)
            if k is None:
                raise ValueError(f"Registry {self.name!r}: 등록 키를 추론할 수 없습니다.")
            if not override and k in self._items:
                raise KeyError(
                    f"Registry {self.name!r}: 키 {k!r} 가 이미 등록되어 있습니다. "
                    f"덮어쓰려면 override=True 사용."
                )
            self._items[k] = obj
            return obj

        return deco

    # --------------------------------------------------------------- lookup/build
    def get(self, key: str) -> T:
        """등록된 객체를 반환한다. 없으면 가용 키 목록과 함께 KeyError."""
        if key not in self._items:
            raise KeyError(
                f"Registry {self.name!r}: 키 {key!r} 가 등록되어 있지 않습니다. "
                f"등록된 키: {sorted(self._items)}"
            )
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def build_with_kwargs(self, key: str, **kwargs: Any) -> Any:
        """등록된 callable/class 를 kwargs 로 호출하여 인스턴스를 만든다."""
        cls_or_fn = self.get(key)
        return cls_or_fn(**kwargs)

    def build(self, spec: Any) -> Any:
        """spec.type / spec.kwargs 규약으로 인스턴스를 만든다.

        Note:
            spec 은 BackboneSpec, FusionSpec 등 ``type: str`` + ``kwargs: dict``
            필드를 가진 dataclass 면 무엇이든 동작한다.
        """
        type_ = getattr(spec, "type", None)
        kwargs = getattr(spec, "kwargs", None)
        if type_ is None:
            raise AttributeError(f"Registry {self.name!r}.build: spec 에 'type' 필드가 없습니다.")
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Registry {self.name!r}.build: spec.kwargs 는 Mapping 이어야 합니다.")
        return self.build_with_kwargs(type_, **dict(kwargs))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry(name={self.name!r}, items={sorted(self._items)})"
