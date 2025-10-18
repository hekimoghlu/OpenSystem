/**
 * GLibUnix 2.0
 *
 * Generated from 2.0
 */

import * as GLib from "glib2";
import * as GObject from "gobject2";

export function closefrom(lowfd: number): number;
export function error_quark(): GLib.Quark;
export function fd_add_full(
    priority: number,
    fd: number,
    condition: GLib.IOCondition,
    _function: GLib.UnixFDSourceFunc
): number;
export function fd_source_new(fd: number, condition: GLib.IOCondition): GLib.Source;
export function fdwalk_set_cloexec(lowfd: number): number;
export function get_passwd_entry(user_name: string): any | null;
export function open_pipe(fds: number, flags: number): boolean;
export function set_fd_nonblocking(fd: number, nonblock: boolean): boolean;
export function signal_add_full(priority: number, signum: number, handler: GLib.SourceFunc): number;
export function signal_source_new(signum: number): GLib.Source;
export type FDSourceFunc = (fd: number, condition: GLib.IOCondition) => boolean;

export namespace PipeEnd {
    export const $gtype: GObject.GType<PipeEnd>;
}

export enum PipeEnd {
    READ = 0,
    WRITE = 1,
}

export class Pipe {
    static $gtype: GObject.GType<Pipe>;

    constructor(copy: Pipe);

    // Fields
    fds: number[];
}
