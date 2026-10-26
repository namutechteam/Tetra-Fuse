"""Registry: name-based object registration, lookup, and build.

Each domain (backbones, fusions, losses, ...) creates its own Registry instance,
and implementation classes/functions are registered via decorator at import time.
Lookup is done by spec.type string, then instantiated with spec.kwargs.

Usage example:
    BACKBONES = Registry("backbones")

    @BACKBONES.register("dmpnn")
    class DMPNNBackbone: ...

    backbone = BACKBONES.build_with_kwargs("dmpnn", depth=3, hidden=300)
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → object mapping container.

    Why:
        To declaratively describe an experiment in YAML, a string
        identifier must be able to specify which object in the code to use. Registry unifies that mapping,
        and kindly reports the list of available keys when an unregistered key is used.

    Attributes:
        name: Domain name for debug messages (e.g., "backbones").
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    # ------------------------------------------------------------------ register
    def register(self, key: str | None = None, *, override: bool = False) -> Callable[[T], T]:
        """Registers an object in decorator form.

        Args:
            key: Registration key. If None, uses ``obj.__name__``.
            override: Whether to allow re-registration of the same key (for testing). Default False.

        Returns:
            A decorator that returns the original object unchanged.
        """

        def deco(obj: T) -> T:
            k = key or getattr(obj, "__name__", None)
            if k is None:
                raise ValueError(f"Registry {self.name!r}: cannot infer the registration key.")
            if not override and k in self._items:
                raise KeyError(
                    f"Registry {self.name!r}: key {k!r} is already registered. "
                    f"Use override=True to overwrite."
                )
            self._items[k] = obj
            return obj

        return deco

    # --------------------------------------------------------------- lookup/build
    def get(self, key: str) -> T:
        """Returns the registered object. Raises KeyError with the list of available keys if not found."""
        if key not in self._items:
            raise KeyError(
                f"Registry {self.name!r}: key {key!r} is not registered. "
                f"Registered keys: {sorted(self._items)}"
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
        """Creates an instance by calling the registered callable/class with kwargs."""
        cls_or_fn = self.get(key)
        return cls_or_fn(**kwargs)

    def build(self, spec: Any) -> Any:
        """Creates an instance using the spec.type / spec.kwargs convention.

        Note:
            spec can be any of BackboneSpec, FusionSpec, etc., with ``type: str`` + ``kwargs: dict``
            fields - any dataclass with those fields works.
        """
        type_ = getattr(spec, "type", None)
        kwargs = getattr(spec, "kwargs", None)
        if type_ is None:
            raise AttributeError(f"Registry {self.name!r}.build: spec has no 'type' field.")
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"Registry {self.name!r}.build: spec.kwargs must be a Mapping.")
        return self.build_with_kwargs(type_, **dict(kwargs))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry(name={self.name!r}, items={sorted(self._items)})"
