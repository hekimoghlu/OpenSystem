/**
 * GstDxva 1.0
 *
 * Generated from 1.0
 */

import * as GstCodecs from "gstcodecs1";
import * as GObject from "gobject2";
import * as Gst from "gst1";
import * as GstVideo from "gstvideo1";

export function dxva_codec_to_string(codec: DxvaCodec): string;

export namespace DxvaCodec {
    export const $gtype: GObject.GType<DxvaCodec>;
}

export enum DxvaCodec {
    NONE = 0,
    MPEG2 = 1,
    H264 = 2,
    H265 = 3,
    VP8 = 4,
    VP9 = 5,
    AV1 = 6,
    LAST = 7,
}
export module DxvaAV1Decoder {
    export interface ConstructorProperties extends GstCodecs.AV1Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaAV1Decoder extends GstCodecs.AV1Decoder {
    static $gtype: GObject.GType<DxvaAV1Decoder>;

    constructor(properties?: Partial<DxvaAV1Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaAV1Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_duplicate_picture(src: GstCodecs.CodecPicture, dst: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.AV1Decoder.vfunc_duplicate_picture
    vfunc_duplicate_picture(...args: never[]): any;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.AV1Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.AV1Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.AV1Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}
export module DxvaH264Decoder {
    export interface ConstructorProperties extends GstCodecs.H264Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaH264Decoder extends GstCodecs.H264Decoder {
    static $gtype: GObject.GType<DxvaH264Decoder>;

    constructor(properties?: Partial<DxvaH264Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaH264Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_duplicate_picture(src: GstCodecs.CodecPicture, dst: GstCodecs.CodecPicture): Gst.FlowReturn;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.H264Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.H264Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.H264Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}
export module DxvaH265Decoder {
    export interface ConstructorProperties extends GstCodecs.H265Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaH265Decoder extends GstCodecs.H265Decoder {
    static $gtype: GObject.GType<DxvaH265Decoder>;

    constructor(properties?: Partial<DxvaH265Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaH265Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.H265Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.H265Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.H265Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}
export module DxvaMpeg2Decoder {
    export interface ConstructorProperties extends GstCodecs.Mpeg2Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaMpeg2Decoder extends GstCodecs.Mpeg2Decoder {
    static $gtype: GObject.GType<DxvaMpeg2Decoder>;

    constructor(properties?: Partial<DxvaMpeg2Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaMpeg2Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    disable_postproc(): void;
    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_duplicate_picture(src: GstCodecs.CodecPicture, dst: GstCodecs.CodecPicture): Gst.FlowReturn;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.Mpeg2Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.Mpeg2Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.Mpeg2Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}
export module DxvaVp8Decoder {
    export interface ConstructorProperties extends GstCodecs.Vp8Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaVp8Decoder extends GstCodecs.Vp8Decoder {
    static $gtype: GObject.GType<DxvaVp8Decoder>;

    constructor(properties?: Partial<DxvaVp8Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaVp8Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp8Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp8Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp8Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}
export module DxvaVp9Decoder {
    export interface ConstructorProperties extends GstCodecs.Vp9Decoder.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class DxvaVp9Decoder extends GstCodecs.Vp9Decoder {
    static $gtype: GObject.GType<DxvaVp9Decoder>;

    constructor(properties?: Partial<DxvaVp9Decoder.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DxvaVp9Decoder.ConstructorProperties>, ...args: any[]): void;

    // Members

    vfunc_configure(
        input_state: GstVideo.VideoCodecState,
        info: GstVideo.VideoInfo,
        crop_x: number,
        crop_y: number,
        coded_width: number,
        coded_height: number,
        max_dpb_size: number
    ): Gst.FlowReturn;
    vfunc_duplicate_picture(src: GstCodecs.CodecPicture, dst: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp9Decoder.vfunc_duplicate_picture
    vfunc_duplicate_picture(...args: never[]): any;
    vfunc_get_picture_id(picture: GstCodecs.CodecPicture): number;
    vfunc_new_picture(picture: GstCodecs.CodecPicture): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp9Decoder.vfunc_new_picture
    vfunc_new_picture(...args: never[]): any;
    vfunc_output_picture(
        frame: GstVideo.VideoCodecFrame,
        picture: GstCodecs.CodecPicture,
        buffer_flags: GstVideo.VideoBufferFlags,
        display_width: number,
        display_height: number
    ): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp9Decoder.vfunc_output_picture
    vfunc_output_picture(...args: never[]): any;
    vfunc_start_picture(picture: GstCodecs.CodecPicture, picture_id: number): Gst.FlowReturn;
    // Conflicted with GstCodecs.Vp9Decoder.vfunc_start_picture
    vfunc_start_picture(...args: never[]): any;
}

export class DxvaAV1DecoderPrivate {
    static $gtype: GObject.GType<DxvaAV1DecoderPrivate>;

    constructor(copy: DxvaAV1DecoderPrivate);
}

export class DxvaDecodingArgs {
    static $gtype: GObject.GType<DxvaDecodingArgs>;

    constructor(
        properties?: Partial<{
            picture_params?: any;
            picture_params_size?: number;
            slice_control?: any;
            slice_control_size?: number;
            bitstream?: any;
            bitstream_size?: number;
            inverse_quantization_matrix?: any;
            inverse_quantization_matrix_size?: number;
        }>
    );
    constructor(copy: DxvaDecodingArgs);

    // Fields
    picture_params: any;
    picture_params_size: number;
    slice_control: any;
    slice_control_size: number;
    bitstream: any;
    bitstream_size: number;
    inverse_quantization_matrix: any;
    inverse_quantization_matrix_size: number;
}

export class DxvaH264DecoderPrivate {
    static $gtype: GObject.GType<DxvaH264DecoderPrivate>;

    constructor(copy: DxvaH264DecoderPrivate);
}

export class DxvaH265DecoderPrivate {
    static $gtype: GObject.GType<DxvaH265DecoderPrivate>;

    constructor(copy: DxvaH265DecoderPrivate);
}

export class DxvaMpeg2DecoderPrivate {
    static $gtype: GObject.GType<DxvaMpeg2DecoderPrivate>;

    constructor(copy: DxvaMpeg2DecoderPrivate);
}

export class DxvaResolution {
    static $gtype: GObject.GType<DxvaResolution>;

    constructor(
        properties?: Partial<{
            width?: number;
            height?: number;
        }>
    );
    constructor(copy: DxvaResolution);

    // Fields
    width: number;
    height: number;
}

export class DxvaVp8DecoderPrivate {
    static $gtype: GObject.GType<DxvaVp8DecoderPrivate>;

    constructor(copy: DxvaVp8DecoderPrivate);
}

export class DxvaVp9DecoderPrivate {
    static $gtype: GObject.GType<DxvaVp9DecoderPrivate>;

    constructor(copy: DxvaVp9DecoderPrivate);
}
