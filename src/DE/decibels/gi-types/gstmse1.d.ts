/**
 * GstMse 1.0
 *
 * Generated from 1.0
 */

import * as Gst from "gst1";
import * as GLib from "glib2";
import * as GObject from "gobject2";

export function media_source_error_quark(): GLib.Quark;

export namespace MediaSourceEOSError {
    export const $gtype: GObject.GType<MediaSourceEOSError>;
}

export enum MediaSourceEOSError {
    NONE = 0,
    NETWORK = 1,
    DECODE = 2,
}

export class MediaSourceError extends GLib.Error {
    static $gtype: GObject.GType<MediaSourceError>;

    constructor(options: { message: string; code: number });
    constructor(copy: MediaSourceError);

    // Fields
    static INVALID_STATE: number;
    static TYPE: number;
    static NOT_SUPPORTED: number;
    static NOT_FOUND: number;
    static QUOTA_EXCEEDED: number;

    // Members
    static quark(): GLib.Quark;
}

export namespace MediaSourceReadyState {
    export const $gtype: GObject.GType<MediaSourceReadyState>;
}

export enum MediaSourceReadyState {
    CLOSED = 0,
    OPEN = 1,
    ENDED = 2,
}

export namespace MseSrcReadyState {
    export const $gtype: GObject.GType<MseSrcReadyState>;
}

export enum MseSrcReadyState {
    NOTHING = 0,
    METADATA = 1,
    CURRENT_DATA = 2,
    FUTURE_DATA = 3,
    ENOUGH_DATA = 4,
}

export namespace SourceBufferAppendMode {
    export const $gtype: GObject.GType<SourceBufferAppendMode>;
}

export enum SourceBufferAppendMode {
    SEGMENTS = 0,
    SEQUENCE = 1,
}
export module MediaSource {
    export interface ConstructorProperties extends Gst.Object.ConstructorProperties {
        [key: string]: any;
        active_source_buffers: SourceBufferList;
        activeSourceBuffers: SourceBufferList;
        duration: number;
        position: number;
        ready_state: MediaSourceReadyState;
        readyState: MediaSourceReadyState;
        source_buffers: SourceBufferList;
        sourceBuffers: SourceBufferList;
    }
}
export class MediaSource extends Gst.Object {
    static $gtype: GObject.GType<MediaSource>;

    constructor(properties?: Partial<MediaSource.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<MediaSource.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get active_source_buffers(): SourceBufferList;
    get activeSourceBuffers(): SourceBufferList;
    get duration(): number;
    set duration(val: number);
    get position(): number;
    set position(val: number);
    get ready_state(): MediaSourceReadyState;
    get readyState(): MediaSourceReadyState;
    get source_buffers(): SourceBufferList;
    get sourceBuffers(): SourceBufferList;

    // Signals

    connect(id: string, callback: (...args: any[]) => any): number;
    connect_after(id: string, callback: (...args: any[]) => any): number;
    emit(id: string, ...args: any[]): void;
    connect(signal: "on-source-close", callback: (_source: this) => void): number;
    connect_after(signal: "on-source-close", callback: (_source: this) => void): number;
    emit(signal: "on-source-close"): void;
    connect(signal: "on-source-ended", callback: (_source: this) => void): number;
    connect_after(signal: "on-source-ended", callback: (_source: this) => void): number;
    emit(signal: "on-source-ended"): void;
    connect(signal: "on-source-open", callback: (_source: this) => void): number;
    connect_after(signal: "on-source-open", callback: (_source: this) => void): number;
    emit(signal: "on-source-open"): void;

    // Constructors

    static ["new"](): MediaSource;

    // Members

    add_source_buffer(type: string): SourceBuffer;
    attach(element: MseSrc): void;
    clear_live_seekable_range(): boolean;
    detach(): void;
    end_of_stream(eos_error: MediaSourceEOSError): boolean;
    get_active_source_buffers(): SourceBufferList;
    get_duration(): Gst.ClockTime;
    get_live_seekable_range(): MediaSourceRange;
    get_position(): Gst.ClockTime;
    get_ready_state(): MediaSourceReadyState;
    get_source_buffers(): SourceBufferList;
    remove_source_buffer(buffer: SourceBuffer): boolean;
    set_duration(duration: Gst.ClockTime): boolean;
    set_live_seekable_range(start: Gst.ClockTime, end: Gst.ClockTime): boolean;
    static is_type_supported(type: string): boolean;
}
export module MseSrc {
    export interface ConstructorProperties extends Gst.Element.ConstructorProperties {
        [key: string]: any;
        duration: number;
        n_audio: number;
        nAudio: number;
        n_text: number;
        nText: number;
        n_video: number;
        nVideo: number;
        position: number;
        ready_state: MseSrcReadyState;
        readyState: MseSrcReadyState;
    }
}
export class MseSrc extends Gst.Element implements Gst.URIHandler {
    static $gtype: GObject.GType<MseSrc>;

    constructor(properties?: Partial<MseSrc.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<MseSrc.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get duration(): number;
    set duration(val: number);
    get n_audio(): number;
    get nAudio(): number;
    get n_text(): number;
    get nText(): number;
    get n_video(): number;
    get nVideo(): number;
    get position(): number;
    get ready_state(): MseSrcReadyState;
    get readyState(): MseSrcReadyState;

    // Members

    get_duration(): Gst.ClockTime;
    get_n_audio(): number;
    get_n_text(): number;
    get_n_video(): number;
    get_position(): Gst.ClockTime;
    get_ready_state(): MseSrcReadyState;

    // Implemented Members

    get_protocols(): string[] | null;
    get_uri(): string | null;
    get_uri_type(): Gst.URIType;
    set_uri(uri: string): boolean;
    vfunc_get_uri(): string | null;
    vfunc_set_uri(uri: string): boolean;
}
export module MseSrcPad {
    export interface ConstructorProperties extends Gst.Pad.ConstructorProperties {
        [key: string]: any;
    }
}
export class MseSrcPad extends Gst.Pad {
    static $gtype: GObject.GType<MseSrcPad>;

    constructor(properties?: Partial<MseSrcPad.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<MseSrcPad.ConstructorProperties>, ...args: any[]): void;
}
export module SourceBuffer {
    export interface ConstructorProperties extends Gst.Object.ConstructorProperties {
        [key: string]: any;
        append_mode: SourceBufferAppendMode;
        appendMode: SourceBufferAppendMode;
        append_window_end: number;
        appendWindowEnd: number;
        append_window_start: number;
        appendWindowStart: number;
        buffered: any[];
        content_type: string;
        contentType: string;
        timestamp_offset: number;
        timestampOffset: number;
        updating: boolean;
    }
}
export class SourceBuffer extends Gst.Object {
    static $gtype: GObject.GType<SourceBuffer>;

    constructor(properties?: Partial<SourceBuffer.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<SourceBuffer.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get append_mode(): SourceBufferAppendMode;
    set append_mode(val: SourceBufferAppendMode);
    get appendMode(): SourceBufferAppendMode;
    set appendMode(val: SourceBufferAppendMode);
    get append_window_end(): number;
    get appendWindowEnd(): number;
    get append_window_start(): number;
    get appendWindowStart(): number;
    get buffered(): any[];
    get content_type(): string;
    set content_type(val: string);
    get contentType(): string;
    set contentType(val: string);
    get timestamp_offset(): number;
    set timestamp_offset(val: number);
    get timestampOffset(): number;
    set timestampOffset(val: number);
    get updating(): boolean;

    // Signals

    connect(id: string, callback: (...args: any[]) => any): number;
    connect_after(id: string, callback: (...args: any[]) => any): number;
    emit(id: string, ...args: any[]): void;
    connect(signal: "on-abort", callback: (_source: this) => void): number;
    connect_after(signal: "on-abort", callback: (_source: this) => void): number;
    emit(signal: "on-abort"): void;
    connect(signal: "on-error", callback: (_source: this) => void): number;
    connect_after(signal: "on-error", callback: (_source: this) => void): number;
    emit(signal: "on-error"): void;
    connect(signal: "on-update", callback: (_source: this) => void): number;
    connect_after(signal: "on-update", callback: (_source: this) => void): number;
    emit(signal: "on-update"): void;
    connect(signal: "on-update-end", callback: (_source: this) => void): number;
    connect_after(signal: "on-update-end", callback: (_source: this) => void): number;
    emit(signal: "on-update-end"): void;
    connect(signal: "on-update-start", callback: (_source: this) => void): number;
    connect_after(signal: "on-update-start", callback: (_source: this) => void): number;
    emit(signal: "on-update-start"): void;

    // Members

    abort(): boolean;
    append_buffer(buf: Gst.Buffer): boolean;
    change_content_type(type: string): boolean;
    get_append_mode(): SourceBufferAppendMode;
    get_append_window_end(): Gst.ClockTime;
    get_append_window_start(): Gst.ClockTime;
    get_buffered(): MediaSourceRange[];
    get_content_type(): string;
    get_timestamp_offset(): Gst.ClockTime;
    get_updating(): boolean;
    remove(start: Gst.ClockTime, end: Gst.ClockTime): boolean;
    set_append_mode(mode: SourceBufferAppendMode): boolean;
    set_append_window_end(end: Gst.ClockTime): boolean;
    set_append_window_start(start: Gst.ClockTime): boolean;
    set_timestamp_offset(offset: Gst.ClockTime): boolean;
}
export module SourceBufferList {
    export interface ConstructorProperties extends Gst.Object.ConstructorProperties {
        [key: string]: any;
        length: number;
    }
}
export class SourceBufferList extends Gst.Object {
    static $gtype: GObject.GType<SourceBufferList>;

    constructor(properties?: Partial<SourceBufferList.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<SourceBufferList.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get length(): number;

    // Signals

    connect(id: string, callback: (...args: any[]) => any): number;
    connect_after(id: string, callback: (...args: any[]) => any): number;
    emit(id: string, ...args: any[]): void;
    connect(signal: "on-sourcebuffer-added", callback: (_source: this) => void): number;
    connect_after(signal: "on-sourcebuffer-added", callback: (_source: this) => void): number;
    emit(signal: "on-sourcebuffer-added"): void;
    connect(signal: "on-sourcebuffer-removed", callback: (_source: this) => void): number;
    connect_after(signal: "on-sourcebuffer-removed", callback: (_source: this) => void): number;
    emit(signal: "on-sourcebuffer-removed"): void;

    // Members

    get_length(): number;
    index(index: number): SourceBuffer | null;
}

export class MediaSourceRange {
    static $gtype: GObject.GType<MediaSourceRange>;

    constructor(copy: MediaSourceRange);

    // Fields
    start: Gst.ClockTime;
    end: Gst.ClockTime;
}

export class SourceBufferInterval {
    static $gtype: GObject.GType<SourceBufferInterval>;

    constructor(copy: SourceBufferInterval);

    // Fields
    start: Gst.ClockTime;
    end: Gst.ClockTime;
}
