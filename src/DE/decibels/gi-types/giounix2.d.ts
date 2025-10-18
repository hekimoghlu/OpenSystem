/**
 * GioUnix 2.0
 *
 * Generated from 2.0
 */

import * as GObject from "gobject2";
import * as Gio from "gio2";
import * as GLib from "glib2";

export const DESKTOP_APP_INFO_LOOKUP_EXTENSION_POINT_NAME: string;
export function desktop_app_info_lookup_get_default_for_uri_scheme(
    lookup: Gio.DesktopAppInfoLookup,
    uri_scheme: string
): Gio.AppInfo | null;
export function file_descriptor_based_get_fd(fd_based: Gio.FileDescriptorBased): number;
export function is_mount_path_system_internal(mount_path: string): boolean;
export function is_system_device_path(device_path: string): boolean;
export function is_system_fs_type(fs_type: string): boolean;
export function mount_at(mount_path: string): [Gio.UnixMountEntry | null, number];
export function mount_compare(mount1: Gio.UnixMountEntry, mount2: Gio.UnixMountEntry): number;
export function mount_copy(mount_entry: Gio.UnixMountEntry): Gio.UnixMountEntry;
export function mount_for(file_path: string): [Gio.UnixMountEntry | null, number];
export function mount_free(mount_entry: Gio.UnixMountEntry): void;
export function mount_get_device_path(mount_entry: Gio.UnixMountEntry): string;
export function mount_get_fs_type(mount_entry: Gio.UnixMountEntry): string;
export function mount_get_mount_path(mount_entry: Gio.UnixMountEntry): string;
export function mount_get_options(mount_entry: Gio.UnixMountEntry): string | null;
export function mount_get_root_path(mount_entry: Gio.UnixMountEntry): string | null;
export function mount_guess_can_eject(mount_entry: Gio.UnixMountEntry): boolean;
export function mount_guess_icon(mount_entry: Gio.UnixMountEntry): Gio.Icon;
export function mount_guess_name(mount_entry: Gio.UnixMountEntry): string;
export function mount_guess_should_display(mount_entry: Gio.UnixMountEntry): boolean;
export function mount_guess_symbolic_icon(mount_entry: Gio.UnixMountEntry): Gio.Icon;
export function mount_is_readonly(mount_entry: Gio.UnixMountEntry): boolean;
export function mount_is_system_internal(mount_entry: Gio.UnixMountEntry): boolean;
export function mount_point_at(mount_path: string): [Gio.UnixMountPoint | null, number];
export function mount_point_compare(mount1: Gio.UnixMountPoint, mount2: Gio.UnixMountPoint): number;
export function mount_point_copy(mount_point: Gio.UnixMountPoint): Gio.UnixMountPoint;
export function mount_point_free(mount_point: Gio.UnixMountPoint): void;
export function mount_point_get_device_path(mount_point: Gio.UnixMountPoint): string;
export function mount_point_get_fs_type(mount_point: Gio.UnixMountPoint): string;
export function mount_point_get_mount_path(mount_point: Gio.UnixMountPoint): string;
export function mount_point_get_options(mount_point: Gio.UnixMountPoint): string | null;
export function mount_point_guess_can_eject(mount_point: Gio.UnixMountPoint): boolean;
export function mount_point_guess_icon(mount_point: Gio.UnixMountPoint): Gio.Icon;
export function mount_point_guess_name(mount_point: Gio.UnixMountPoint): string;
export function mount_point_guess_symbolic_icon(mount_point: Gio.UnixMountPoint): Gio.Icon;
export function mount_point_is_loopback(mount_point: Gio.UnixMountPoint): boolean;
export function mount_point_is_readonly(mount_point: Gio.UnixMountPoint): boolean;
export function mount_point_is_user_mountable(mount_point: Gio.UnixMountPoint): boolean;
export function mount_points_changed_since(time: number): boolean;
export function mount_points_get(): [Gio.UnixMountPoint[], number];
export function mounts_changed_since(time: number): boolean;
export function mounts_get(): [Gio.UnixMountEntry[], number];
export type DesktopAppLaunchCallback = (appinfo: Gio.DesktopAppInfo, pid: GLib.Pid) => void;
export module DesktopAppInfo {
    export interface ConstructorProperties extends GObject.Object.ConstructorProperties {
        [key: string]: any;
        filename: string;
    }
}
export class DesktopAppInfo extends GObject.Object implements Gio.AppInfo {
    static $gtype: GObject.GType<DesktopAppInfo>;

    constructor(properties?: Partial<DesktopAppInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DesktopAppInfo.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get filename(): string;

    // Constructors

    static ["new"](desktop_id: string): DesktopAppInfo;
    static new_from_filename(filename: string): DesktopAppInfo;
    static new_from_keyfile(key_file: GLib.KeyFile): DesktopAppInfo;

    // Members

    static get_action_name(info: Gio.DesktopAppInfo, action_name: string): string;
    static get_boolean(info: Gio.DesktopAppInfo, key: string): boolean;
    static get_categories(info: Gio.DesktopAppInfo): string | null;
    static get_filename(info: Gio.DesktopAppInfo): string | null;
    static get_generic_name(info: Gio.DesktopAppInfo): string | null;
    static get_implementations(_interface: string): Gio.DesktopAppInfo[];
    static get_is_hidden(info: Gio.DesktopAppInfo): boolean;
    static get_keywords(info: Gio.DesktopAppInfo): string[];
    static get_locale_string(info: Gio.DesktopAppInfo, key: string): string | null;
    static get_nodisplay(info: Gio.DesktopAppInfo): boolean;
    static get_show_in(info: Gio.DesktopAppInfo, desktop_env?: string | null): boolean;
    static get_startup_wm_class(info: Gio.DesktopAppInfo): string | null;
    static get_string(info: Gio.DesktopAppInfo, key: string): string | null;
    static get_string_list(info: Gio.DesktopAppInfo, key: string): string[];
    static has_key(info: Gio.DesktopAppInfo, key: string): boolean;
    static launch_action(
        info: Gio.DesktopAppInfo,
        action_name: string,
        launch_context?: Gio.AppLaunchContext | null
    ): void;
    static launch_uris_as_manager(
        appinfo: Gio.DesktopAppInfo,
        uris: string[],
        launch_context: Gio.AppLaunchContext | null,
        spawn_flags: GLib.SpawnFlags,
        user_setup?: GLib.SpawnChildSetupFunc | null,
        pid_callback?: Gio.DesktopAppLaunchCallback | null
    ): boolean;
    static launch_uris_as_manager_with_fds(
        appinfo: Gio.DesktopAppInfo,
        uris: string[],
        launch_context: Gio.AppLaunchContext | null,
        spawn_flags: GLib.SpawnFlags,
        user_setup: GLib.SpawnChildSetupFunc | null,
        pid_callback: Gio.DesktopAppLaunchCallback | null,
        stdin_fd: number,
        stdout_fd: number,
        stderr_fd: number
    ): boolean;
    static list_actions(info: Gio.DesktopAppInfo): string[];
    static search(search_string: string): string[][];
    static set_desktop_env(desktop_env: string): void;

    // Implemented Members

    add_supports_type(content_type: string): boolean;
    can_delete(): boolean;
    can_remove_supports_type(): boolean;
    ["delete"](): boolean;
    dup(): Gio.AppInfo;
    equal(appinfo2: Gio.AppInfo): boolean;
    get_commandline(): string | null;
    get_description(): string | null;
    get_display_name(): string;
    get_executable(): string;
    get_icon(): Gio.Icon | null;
    get_id(): string | null;
    get_name(): string;
    get_supported_types(): string[];
    launch(files?: Gio.File[] | null, context?: Gio.AppLaunchContext | null): boolean;
    launch_uris(uris?: string[] | null, context?: Gio.AppLaunchContext | null): boolean;
    launch_uris_async(
        uris?: string[] | null,
        context?: Gio.AppLaunchContext | null,
        cancellable?: Gio.Cancellable | null
    ): Promise<boolean>;
    launch_uris_async(
        uris: string[] | null,
        context: Gio.AppLaunchContext | null,
        cancellable: Gio.Cancellable | null,
        callback: Gio.AsyncReadyCallback<this> | null
    ): void;
    launch_uris_async(
        uris?: string[] | null,
        context?: Gio.AppLaunchContext | null,
        cancellable?: Gio.Cancellable | null,
        callback?: Gio.AsyncReadyCallback<this> | null
    ): Promise<boolean> | void;
    launch_uris_finish(result: Gio.AsyncResult): boolean;
    remove_supports_type(content_type: string): boolean;
    set_as_default_for_extension(extension: string): boolean;
    set_as_default_for_type(content_type: string): boolean;
    set_as_last_used_for_type(content_type: string): boolean;
    should_show(): boolean;
    supports_files(): boolean;
    supports_uris(): boolean;
    vfunc_add_supports_type(content_type: string): boolean;
    vfunc_can_delete(): boolean;
    vfunc_can_remove_supports_type(): boolean;
    vfunc_do_delete(): boolean;
    vfunc_dup(): Gio.AppInfo;
    vfunc_equal(appinfo2: Gio.AppInfo): boolean;
    vfunc_get_commandline(): string | null;
    vfunc_get_description(): string | null;
    vfunc_get_display_name(): string;
    vfunc_get_executable(): string;
    vfunc_get_icon(): Gio.Icon | null;
    vfunc_get_id(): string | null;
    vfunc_get_name(): string;
    vfunc_get_supported_types(): string[];
    vfunc_launch(files?: Gio.File[] | null, context?: Gio.AppLaunchContext | null): boolean;
    vfunc_launch_uris(uris?: string[] | null, context?: Gio.AppLaunchContext | null): boolean;
    vfunc_launch_uris_async(
        uris?: string[] | null,
        context?: Gio.AppLaunchContext | null,
        cancellable?: Gio.Cancellable | null
    ): Promise<boolean>;
    vfunc_launch_uris_async(
        uris: string[] | null,
        context: Gio.AppLaunchContext | null,
        cancellable: Gio.Cancellable | null,
        callback: Gio.AsyncReadyCallback<this> | null
    ): void;
    vfunc_launch_uris_async(
        uris?: string[] | null,
        context?: Gio.AppLaunchContext | null,
        cancellable?: Gio.Cancellable | null,
        callback?: Gio.AsyncReadyCallback<this> | null
    ): Promise<boolean> | void;
    vfunc_launch_uris_finish(result: Gio.AsyncResult): boolean;
    vfunc_remove_supports_type(content_type: string): boolean;
    vfunc_set_as_default_for_extension(extension: string): boolean;
    vfunc_set_as_default_for_type(content_type: string): boolean;
    vfunc_set_as_last_used_for_type(content_type: string): boolean;
    vfunc_should_show(): boolean;
    vfunc_supports_files(): boolean;
    vfunc_supports_uris(): boolean;
}
export module FDMessage {
    export interface ConstructorProperties extends Gio.SocketControlMessage.ConstructorProperties {
        [key: string]: any;
        fd_list: Gio.UnixFDList;
        fdList: Gio.UnixFDList;
    }
}
export class FDMessage extends Gio.SocketControlMessage {
    static $gtype: GObject.GType<FDMessage>;

    constructor(properties?: Partial<FDMessage.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<FDMessage.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get fd_list(): Gio.UnixFDList;
    get fdList(): Gio.UnixFDList;

    // Fields
    priv: Gio.UnixFDMessagePrivate | any;

    // Constructors

    static ["new"](): FDMessage;
    static new_with_fd_list(fd_list: Gio.UnixFDList): FDMessage;

    // Members

    static append_fd(message: Gio.UnixFDMessage, fd: number): boolean;
    static get_fd_list(message: Gio.UnixFDMessage): Gio.UnixFDList;
    static steal_fds(message: Gio.UnixFDMessage): number[];
}
export module InputStream {
    export interface ConstructorProperties extends Gio.InputStream.ConstructorProperties {
        [key: string]: any;
        close_fd: boolean;
        closeFd: boolean;
        fd: number;
    }
}
export class InputStream extends Gio.InputStream implements Gio.PollableInputStream, FileDescriptorBased {
    static $gtype: GObject.GType<InputStream>;

    constructor(properties?: Partial<InputStream.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<InputStream.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get close_fd(): boolean;
    set close_fd(val: boolean);
    get closeFd(): boolean;
    set closeFd(val: boolean);
    get fd(): number;

    // Constructors

    static ["new"](fd: number, close_fd: boolean): InputStream;

    // Members

    static get_close_fd(stream: Gio.UnixInputStream): boolean;
    static get_fd(stream: Gio.UnixInputStream): number;
    static set_close_fd(stream: Gio.UnixInputStream, close_fd: boolean): void;

    // Implemented Members

    can_poll(): boolean;
    create_source(cancellable?: Gio.Cancellable | null): GLib.Source;
    is_readable(): boolean;
    read_nonblocking(cancellable?: Gio.Cancellable | null): [number, Uint8Array];
    vfunc_can_poll(): boolean;
    vfunc_create_source(cancellable?: Gio.Cancellable | null): GLib.Source;
    vfunc_is_readable(): boolean;
    vfunc_read_nonblocking(): [number, Uint8Array | null];
}
export module MountMonitor {
    export interface ConstructorProperties extends GObject.Object.ConstructorProperties {
        [key: string]: any;
    }
}
export class MountMonitor extends GObject.Object {
    static $gtype: GObject.GType<MountMonitor>;

    constructor(properties?: Partial<MountMonitor.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<MountMonitor.ConstructorProperties>, ...args: any[]): void;

    // Signals

    connect(id: string, callback: (...args: any[]) => any): number;
    connect_after(id: string, callback: (...args: any[]) => any): number;
    emit(id: string, ...args: any[]): void;
    connect(signal: "mountpoints-changed", callback: (_source: this) => void): number;
    connect_after(signal: "mountpoints-changed", callback: (_source: this) => void): number;
    emit(signal: "mountpoints-changed"): void;
    connect(signal: "mounts-changed", callback: (_source: this) => void): number;
    connect_after(signal: "mounts-changed", callback: (_source: this) => void): number;
    emit(signal: "mounts-changed"): void;

    // Constructors

    static ["new"](): MountMonitor;

    // Members

    static get(): Gio.UnixMountMonitor;
    static set_rate_limit(mount_monitor: Gio.UnixMountMonitor, limit_msec: number): void;
}
export module OutputStream {
    export interface ConstructorProperties extends Gio.OutputStream.ConstructorProperties {
        [key: string]: any;
        close_fd: boolean;
        closeFd: boolean;
        fd: number;
    }
}
export class OutputStream extends Gio.OutputStream implements Gio.PollableOutputStream, FileDescriptorBased {
    static $gtype: GObject.GType<OutputStream>;

    constructor(properties?: Partial<OutputStream.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<OutputStream.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get close_fd(): boolean;
    set close_fd(val: boolean);
    get closeFd(): boolean;
    set closeFd(val: boolean);
    get fd(): number;

    // Constructors

    static ["new"](fd: number, close_fd: boolean): OutputStream;

    // Members

    static get_close_fd(stream: Gio.UnixOutputStream): boolean;
    static get_fd(stream: Gio.UnixOutputStream): number;
    static set_close_fd(stream: Gio.UnixOutputStream, close_fd: boolean): void;

    // Implemented Members

    can_poll(): boolean;
    create_source(cancellable?: Gio.Cancellable | null): GLib.Source;
    is_writable(): boolean;
    write_nonblocking(buffer: Uint8Array | string, cancellable?: Gio.Cancellable | null): number;
    writev_nonblocking(vectors: Gio.OutputVector[], cancellable?: Gio.Cancellable | null): [Gio.PollableReturn, number];
    vfunc_can_poll(): boolean;
    vfunc_create_source(cancellable?: Gio.Cancellable | null): GLib.Source;
    vfunc_is_writable(): boolean;
    vfunc_write_nonblocking(buffer?: Uint8Array | null): number;
    vfunc_writev_nonblocking(vectors: Gio.OutputVector[]): [Gio.PollableReturn, number];
}

export class FDMessagePrivate {
    static $gtype: GObject.GType<FDMessagePrivate>;

    constructor(copy: FDMessagePrivate);
}

export class InputStreamPrivate {
    static $gtype: GObject.GType<InputStreamPrivate>;

    constructor(copy: InputStreamPrivate);
}

export class MountEntry {
    static $gtype: GObject.GType<MountEntry>;

    constructor(copy: MountEntry);
}

export class MountPoint {
    static $gtype: GObject.GType<MountPoint>;

    constructor(copy: MountPoint);

    // Members
    static at(mount_path: string): [Gio.UnixMountPoint | null, number];
    static compare(mount1: Gio.UnixMountPoint, mount2: Gio.UnixMountPoint): number;
    static copy(mount_point: Gio.UnixMountPoint): Gio.UnixMountPoint;
    static free(mount_point: Gio.UnixMountPoint): void;
    static get_device_path(mount_point: Gio.UnixMountPoint): string;
    static get_fs_type(mount_point: Gio.UnixMountPoint): string;
    static get_mount_path(mount_point: Gio.UnixMountPoint): string;
    static get_options(mount_point: Gio.UnixMountPoint): string | null;
    static guess_can_eject(mount_point: Gio.UnixMountPoint): boolean;
    static guess_icon(mount_point: Gio.UnixMountPoint): Gio.Icon;
    static guess_name(mount_point: Gio.UnixMountPoint): string;
    static guess_symbolic_icon(mount_point: Gio.UnixMountPoint): Gio.Icon;
    static is_loopback(mount_point: Gio.UnixMountPoint): boolean;
    static is_readonly(mount_point: Gio.UnixMountPoint): boolean;
    static is_user_mountable(mount_point: Gio.UnixMountPoint): boolean;
}

export class OutputStreamPrivate {
    static $gtype: GObject.GType<OutputStreamPrivate>;

    constructor(copy: OutputStreamPrivate);
}

export interface DesktopAppInfoLookupNamespace {
    $gtype: GObject.GType<DesktopAppInfoLookup>;
    prototype: DesktopAppInfoLookupPrototype;

    get_default_for_uri_scheme(lookup: Gio.DesktopAppInfoLookup, uri_scheme: string): Gio.AppInfo | null;
}
export type DesktopAppInfoLookup = DesktopAppInfoLookupPrototype;
export interface DesktopAppInfoLookupPrototype extends GObject.Object {}

export const DesktopAppInfoLookup: DesktopAppInfoLookupNamespace;

export interface FileDescriptorBasedNamespace {
    $gtype: GObject.GType<FileDescriptorBased>;
    prototype: FileDescriptorBasedPrototype;

    get_fd(fd_based: Gio.FileDescriptorBased): number;
}
export type FileDescriptorBased = FileDescriptorBasedPrototype;
export interface FileDescriptorBasedPrototype extends GObject.Object {}

export const FileDescriptorBased: FileDescriptorBasedNamespace;
