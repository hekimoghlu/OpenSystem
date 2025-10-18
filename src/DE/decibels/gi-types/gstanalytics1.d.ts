/**
 * GstAnalytics 1.0
 *
 * Generated from 1.0
 */

import * as Gst from "gst1";
import * as GObject from "gobject2";
import * as GLib from "glib2";

export const INF_RELATION_SPAN: number;
export const MTD_TYPE_ANY: number;
export function buffer_add_analytics_relation_meta(buffer: Gst.Buffer): RelationMeta | null;
export function buffer_add_analytics_relation_meta_full(
    buffer: Gst.Buffer,
    init_params: RelationMetaInitParams
): RelationMeta | null;
export function buffer_get_analytics_relation_meta(buffer: Gst.Buffer): RelationMeta | null;
export function cls_mtd_get_mtd_type(): MtdType;
export function mtd_type_get_name(type: MtdType): string;
export function od_mtd_get_mtd_type(): MtdType;
export function relation_get_length(instance: RelationMeta): number;
export function relation_meta_api_get_type(): GObject.GType;
export function tracking_mtd_get_mtd_type(): MtdType;

export namespace RelTypes {
    export const $gtype: GObject.GType<RelTypes>;
}

export enum RelTypes {
    NONE = 0,
    IS_PART_OF = 2,
    CONTAIN = 4,
    RELATE_TO = 8,
    LAST = 16,
    ANY = 2147483647,
}

export class ClsMtd {
    static $gtype: GObject.GType<ClsMtd>;

    constructor(copy: ClsMtd);

    // Fields
    id: number;
    meta: RelationMeta;

    // Members
    get_index_by_quark(quark: GLib.Quark): number;
    get_length(): number;
    get_level(index: number): number;
    get_quark(index: number): GLib.Quark;
    static get_mtd_type(): MtdType;
}

export class Mtd {
    static $gtype: GObject.GType<Mtd>;

    constructor(copy: Mtd);

    // Fields
    id: number;
    meta: RelationMeta;

    // Members
    get_id(): number;
    get_mtd_type(): MtdType;
    get_size(): number;
    static type_get_name(type: MtdType): string;
}

export class MtdImpl {
    static $gtype: GObject.GType<MtdImpl>;

    constructor(
        properties?: Partial<{
            name?: string;
        }>
    );
    constructor(copy: MtdImpl);

    // Fields
    name: string;
}

export class ODMtd {
    static $gtype: GObject.GType<ODMtd>;

    constructor(copy: ODMtd);

    // Fields
    id: number;
    meta: RelationMeta;

    // Members
    get_confidence_lvl(): [boolean, number];
    get_location(): [boolean, number, number, number, number, number];
    get_obj_type(): GLib.Quark;
    static get_mtd_type(): MtdType;
}

export class RelationMeta {
    static $gtype: GObject.GType<RelationMeta>;

    constructor(copy: RelationMeta);

    // Members
    add_cls_mtd(confidence_levels: number[], class_quarks: GLib.Quark[]): [boolean, ClsMtd];
    add_od_mtd(
        type: GLib.Quark,
        x: number,
        y: number,
        w: number,
        h: number,
        loc_conf_lvl: number
    ): [boolean, ODMtd | null];
    add_one_cls_mtd(confidence_level: number, class_quark: GLib.Quark): [boolean, ClsMtd];
    add_tracking_mtd(tracking_id: number, tracking_first_seen: Gst.ClockTime): [boolean, TrackingMtd];
    exist(
        an_meta_first_id: number,
        an_meta_second_id: number,
        max_relation_span: number,
        cond_types: RelTypes
    ): [boolean, number[] | null];
    get_cls_mtd(an_meta_id: number): [boolean, ClsMtd];
    get_direct_related(
        an_meta_id: number,
        relation_type: RelTypes,
        type: MtdType,
        state: any,
        rlt_mtd: Mtd
    ): [boolean, any];
    get_mtd(an_meta_id: number, type: MtdType): [boolean, Mtd];
    get_od_mtd(an_meta_id: number): [boolean, ODMtd];
    get_relation(an_meta_first_id: number, an_meta_second_id: number): RelTypes;
    get_tracking_mtd(an_meta_id: number): [boolean, TrackingMtd];
    iterate(state: any | null, type: MtdType, rlt_mtd: Mtd): boolean;
    set_relation(type: RelTypes, an_meta_first_id: number, an_meta_second_id: number): boolean;
}

export class RelationMetaInitParams {
    static $gtype: GObject.GType<RelationMetaInitParams>;

    constructor(
        properties?: Partial<{
            initial_relation_order?: number;
            initial_buf_size?: number;
        }>
    );
    constructor(copy: RelationMetaInitParams);

    // Fields
    initial_relation_order: number;
    initial_buf_size: number;
}

export class TrackingMtd {
    static $gtype: GObject.GType<TrackingMtd>;

    constructor(copy: TrackingMtd);

    // Fields
    id: number;
    meta: RelationMeta;

    // Members
    get_info(): [boolean, number, Gst.ClockTime, Gst.ClockTime, boolean];
    set_lost(): boolean;
    update_last_seen(last_seen: Gst.ClockTime): boolean;
    static get_mtd_type(): MtdType;
}
export type MtdType = never;
