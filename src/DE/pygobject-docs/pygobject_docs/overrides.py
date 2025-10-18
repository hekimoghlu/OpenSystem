from typing import Any, Callable, Sequence

from gi.repository import GLib, GObject


# GLib
def gi__gi_add_emission_hook(
    type: GObject.Object, name: str, callback: Callable[[...], None], *args: Any  # type: ignore[misc]
) -> None:
    ...


def gi__gi_spawn_async(  # type: ignore[empty-body]
    argv: Sequence[str],
    envp: Sequence[str] | None = None,
    working_directory: str | None = None,
    flags: GLib.SpawnFlags = GLib.SpawnFlags.DEFAULT,
    child_setup: Callable[[Any], None] | None = None,
    user_data: Any = None,
    standard_input: bool = False,
    standard_output: bool = False,
    standard_error: bool = False,
) -> tuple[GLib.Pid, int | None, int | None, int | None]:
    """
    Execute a child program asynchronously within a GLib main loop.
    See the reference manual for a complete reference.
    """


def gi__gi_Pid_close() -> None:
    ...


# GObject
def gi__gi_list_properties() -> list[GObject.ParamSpec]:  # type: ignore[empty-body]
    ...


def gi__gi_new(gtype: GObject.GType) -> None:
    ...


def gi__gi_signal_new(  # type: ignore[empty-body]
    signal_name: str,
    itype: type[GObject.Object],
    signal_flags: GObject.SignalFlags,
    return_type: type,
    param_types: Sequence[GObject.GType],
) -> int:
    ...


def gi__gi_type_register(type) -> GObject.GType:
    ...


def gi__gi_GObject___init__(**properties: Any):
    ...


def gi__gi_GObject_bind_property(
    source_property: str,
    target: GObject.Object,
    target_property: str,
    flags: GObject.BindingFlags | None = 0,
    transform_to: Callable[[GObject.Binding, Any, Any], Any] | None = None,
    transform_from: Callable[[GObject.Binding, Any, Any], Any] | None = None,
    user_data: Any = None,
) -> GObject.Binding:
    """
    Creates a binding between a property on the source object and
    a property on the target object.

    :param source_prop:
        The property name on the source object
    :param target:
        The target object
    :param target_prop:
        The property name on the target
    :param flags:
        Optional flags to pass to the binding. Defaults to :ref:`GObject.BindingFlags.DEFAULT`
    :param transform_to:
        Optional transformation function from the source to the target.
        The second argument of the callable is the source value.
        The third argument is the user data.
        This function should return the value to be set on the target, with the right type.
    :param transform_from:
        Optional transformation function from the target to the source
        The second argument of the callable is the target value.
        The third argument is the user data.
        This function should return the value to be set on the source, with the right type.
    :param user_data:
        Optional user data, provided as third argument to the transformation functions.
    :returns:
        A new :obj:`~gi.repository.GObject.Binding` object, representing the binding.
    """


def gi__gi_GObject_chain(*params) -> object | None:
    """
    Calls the original class closure of a signal.

    This function should only be called from an overridden class closure.

    :param params: The argument list of the signal emission.
    :returns: Return value of the signal handler.
    """


def gi__gi_GObject_connect(  # type: ignore[empty-body]
    detailed_signal: str, handler: Callable[[GObject.Object, ...], Any], *args: Any  # type: ignore[misc]
) -> int:
    """
    The ``connect()`` method adds a function or method (handler) to the end of the list of signal handlers for the named ``detailed_signal`` but before the default class signal handler. An optional set of parameters may be specified after the handler parameter. These will all be passed to the signal handler when invoked.

    For example if a function handler was connected to a signal using:

    .. code:: python

        handler_id = object.connect("signal_name", handler, arg1, arg2, arg3)

    The handler should be defined as:

    .. code:: python

        def handler(object, arg1, arg2, arg3):

    A method handler connected to a signal using:

    .. code:: python

        handler_id = object.connect("signal_name", self.handler, arg1, arg2)

    requires an additional argument when defined:

    .. code:: python

        def handler(self, object, arg1, arg2)

    A :ref:`TypeError` exception is raised if ``detailed_signal`` identifies a signal name that is not associated with the object.

    :param detailed_signal: A signal name.
    :param handler: Callback to invoke if the signal is emitted. The callback signature needs to match the signature of the signal.
    :param args: Additional arguments to pass to the callback.
    """


def gi__gi_GObject_connect_after(  # type: ignore[empty-body]
    detailed_signal: str, handler: Callable[[GObject.Object, ...], Any], *args: Any  # type: ignore[misc]
) -> int:
    """
    The ``connect_after()`` method is similar to the :meth:`~gi.repository.GObject.Object.connect` method except that the handler is added to the signal handler list after the default class signal handler. Otherwise the details of handler definition and invocation are the same.

    :param detailed_signal: A signal name.
    :param handler: Callback to invoke if the signal is emitted. The callback signature needs to match the signature of the signal.
    :param args: Additional arguments to pass to the callback.
    """


def gi__gi_GObject_connect_object(  # type: ignore[empty-body]
    detailed_signal: str, handler: Callable[[GObject.Object, ...], Any], object: GObject.Object, *args: Any  # type: ignore[misc]
) -> int:
    """
    The ``connect_after()`` method is similar to the :meth:`~gi.repository.GObject.Object.connect` method except that it takes an additional object as argument. The object is weakly referenced and the signal is
    automatically disconnected when the object is finalized.
    """


def gi__gi_GObject_connect_object_after(  # type: ignore[empty-body]
    detailed_signal: str, handler: Callable[[GObject.Object, Any], Any], object: GObject.Object, *args: Any
) -> int:
    """
    The ``connect_object_after()`` method is similar to the :meth:`~gi.repository.GObject.Object.connect_object` method except that the handler is added to the signal handler list after the default class signal handler. Otherwise the details of handler definition and invocation are the same.
    """


def gi__gi_GObject_disconnect_by_func(func: Callable[[GObject.Object, ...], Any]) -> None:  # type: ignore[misc]
    """
    Disconnect a function (callable) from any signal.
    """


def gi__gi_GObject_emit(signal_name: str, *args) -> None:
    ...


def gi__gi_GObject_get_properties(*prop_names: str) -> tuple[Any, ...]:  # type: ignore[empty-body]
    ...


def gi__gi_GObject_get_property(prop_name: str) -> Any:
    ...


def gi__gi_GObject_handler_block_by_func(func: Callable[[GObject.Object, ...], ...]) -> int:  # type: ignore[empty-body,misc]
    ...


def gi__gi_GObject_handler_unblock_by_func(func: Callable[[GObject.Object, ...], ...]) -> int:  # type: ignore[empty-body,misc]
    ...


def gi__gi_GObject_set_properties(**props) -> None:
    ...


def gi__gi_GObject_set_property(prop_name: str, prop_value: Any) -> None:
    ...


def gi__gi_GObject_weak_ref(callback: Callable[[Any], None] | None, *args: Any) -> GObject.Object:
    ...


# GLib.OptionContext


def gi__gi_OptionContext_add_group(group: GLib.OptionGroup) -> None:
    ...


def gi__gi_OptionContext_get_help_enabled() -> bool:  # type: ignore[empty-body]
    ...


def gi__gi_OptionContext_get_ignore_unknown_options() -> bool:  # type: ignore[empty-body]
    ...


def gi__gi_OptionContext_get_main_group() -> GLib.OptionGroup:
    ...


def gi__gi_OptionContext_parse(argv: Sequence[str]) -> tuple[bool, list[str]]:  # type: ignore[empty-body]
    ...


def gi__gi_OptionContext_set_help_enabled(help_enabled: bool) -> None:
    ...


def gi__gi_OptionContext_set_ignore_unknown_options(ignore_unknown: bool) -> None:
    ...


def gi__gi_OptionContext_set_main_group(group: GLib.OptionGroup) -> None:
    ...


def gi__gi_OptionGroup_add_entries(entries: list[GLib.OptionEntry]) -> None:
    ...


def gi__gi_OptionGroup_set_translation_domain(domain: str) -> None:
    ...


def gi__gi_GObjectWeakRef_unref() -> None:
    ...


def gobject_GBoxed_copy() -> GObject.GBoxed:
    ...


# GObject.GType


def None_from_name(name: str) -> GObject.GType:
    ...


def gobject_GType_has_value_table() -> None:
    ...


def gobject_GType_is_a(type: GObject.GType) -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_abstract() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_classed() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_deep_derivable() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_derivable() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_instantiatable() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_interface() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_value_abstract() -> bool:  # type: ignore[empty-body]
    ...


def gobject_GType_is_value_type() -> bool:  # type: ignore[empty-body]
    ...


# gi.repository.GLib


class gi_repository_GLib_Error:
    """The ``Error`` structure contains information about an error that has occurred.

    Attributes
    ----------

    .. rst-class:: interim-class

    .. class:: Error
        :no-index:

        .. attribute:: domain

            The error domain, usually a string that you can convert to a
            GLib quark with :func:`~gi.repository.GLib.quark_from_string`.

        .. attribute:: code

            A numeric code that identifies a specific error within the domain.

        .. attribute:: message

            A human-readable description of the error.
    """
