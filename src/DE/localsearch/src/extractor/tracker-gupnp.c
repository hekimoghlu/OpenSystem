/*
 * Copyright (C) 2025 Red Hat Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-gupnp.h"

#include <libavcodec/codec_desc.h>

struct CodecProps {
	enum AVCodecID id;
	const char *mime_type;
	const char *variant;
	const char *stream_format;
	const char *profile;
	int mpegversion;
	int wmaversion;
	int layer;
	gboolean systemstream;
	gboolean interlaced;
} codec_map[] = {
	{ AV_CODEC_ID_MPEG1VIDEO, "video/mpeg", NULL, NULL, NULL, 1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MPEG2VIDEO, "video/mpeg", NULL, NULL, NULL, 2, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_H261, "video/x-h261",  NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_H263, "video/x-h263", "itu", NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_H263P, "video/x-h263", "itu", NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_H263I, "video/x-intel-h263", "intel", NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_H264, "video/x-h264", NULL, "avc", NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_HEVC, "video/x-h265", NULL, "hvc1", NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MJPEG, "video/x-mjpeg" },
	{ AV_CODEC_ID_MPEG4, "video/mpeg", NULL, NULL, NULL, 4, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MSMPEG4V1, "video/x-msmpeg", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MSMPEG4V2, "video/x-msmpeg", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MSMPEG4V3, "video/x-msmpeg", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMV1, "video/x-wmv", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMV2, "video/x-wmv", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_FLV1, "video/x-flash-video", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_THEORA, "video/x-theora", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_4XM, "video/x-4xm", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MSVIDEO1, "video/x-msvideocodec", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_AMR_NB, "audio/AMR", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_AMR_WB, "audio/AMR-WB", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_MP1, "audio/mpeg", NULL, NULL, NULL, 1, -1, 1, FALSE, FALSE },
	{ AV_CODEC_ID_MP2, "audio/mpeg", NULL, NULL, NULL, 1, -1, 2, FALSE, FALSE },
	{ AV_CODEC_ID_MP3, "audio/mpeg", NULL, NULL, NULL, 1, -1, 3, FALSE, FALSE },
	{ AV_CODEC_ID_AAC, "audio/mpeg", NULL, "raw",  "lc", 4, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_AAC_LATM, "audio/mpeg", NULL, "loas", NULL, 4, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_AC3, "audio/x-ac3", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_DTS, "audio/x-dts", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_VORBIS, "audio/x-vorbis", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMAV1, "audio/x-wma", NULL, NULL, NULL, -1, 1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMAV2, "audio/x-wma", NULL, NULL, NULL, -1, 2, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMAPRO, "audio/x-wma", NULL, NULL, NULL, -1, 3, -1, FALSE, FALSE },
	{ AV_CODEC_ID_WMALOSSLESS, "audio/x-wma", NULL, NULL, NULL, -1, 4, -1, FALSE, FALSE },
	{ AV_CODEC_ID_FLAC, "audio/x-flac", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_GSM, "audio/x-gsm", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
	{ AV_CODEC_ID_EAC3, "audio/x-eac3", NULL, NULL, NULL, -1, -1, -1, FALSE, FALSE },
};

struct ContainerProps {
	const gchar *format;
	const gchar *mime_type;
	const gchar *variant;
	int mpegversion;
	gboolean systemstream;
} container_map[] = {
	{ "mpeg", "video/mpeg", NULL, -1, TRUE },
	{ "mpegts", "video/mpegts", NULL, -1, TRUE },
	{ "rm", "application/x-pn-realmedia", NULL, -1, TRUE },
	{ "asf", "video/x-ms-asf", NULL, -1, FALSE },
	{ "avi", "video/x-msvideo", NULL, -1, FALSE },
	{ "wav", "audio/x-wav", NULL, -1, FALSE },
	{ "ape", "application/x-ape", NULL, -1, FALSE },
	{ "swf", "application/x-shockwave-flash", NULL, -1, FALSE },
	{ "au", "audo/x-au", NULL, -1, FALSE },
	{ "dv", "video/x-dv", NULL, -1, TRUE },
	{ "4xm", "video/x-4xm", NULL, -1, FALSE },
	{ "matroska", "video/x-matroska", NULL, -1, FALSE },
	{ "ivf", "video/x-ivf", NULL, -1, FALSE },
	{ "mp3", "applications/x-id3", NULL, -1, FALSE },
	{ "flic", "video/x-fli", NULL, -1, FALSE },
	{ "flv", "video/x-flv", NULL, -1, FALSE },
	{ "tta", "audio/x-ttafile", NULL, -1, FALSE },
	{ "aiff", "audio/x-aiff", NULL, -1, FALSE },
	{ "mov,mp4,m4a,3gp,3g2,mj2", "video/quicktime", "iso", -1, FALSE },
	{ "mov", "video/quicktime", NULL, -1, FALSE },
	{ "mp4", "video/quicktime", "iso", -1, FALSE },
	{ "3gp", "video/quicktime", "3gpp", -1, FALSE },
	{ "3g2", "video/quicktime", "3g2", -1, FALSE },
	{ "psp", "video/quicktime", "psp", -1, FALSE },
	{ "ipod", "video/quicktime", "ipod", -1, FALSE },
	{ "aac", "audio/mpeg", NULL, 4, FALSE },
	{ "ogg", "application/ogg", NULL, -1, FALSE },
	{ "mxf", "application/mxf", NULL, -1, FALSE },
	{ "mxf_d10", "application/mxf", NULL, -1, FALSE },
	{ "gxf", "application/gxf", NULL, -1, FALSE },
	{ "yuv4mpegpipe", "application/x-yuv4mpeg", NULL, -1, FALSE },
	{ "mpc", "audio/x-musepack", NULL, -1, FALSE },
	{ "mpc8", "audio/x-musepack", NULL, -1, FALSE },
	{ "vqf", "audio/x-vqf", NULL, -1, FALSE },
	{ "nsv", "video/x-nsv", NULL, -1, FALSE },
	{ "amr", "audio/x-amr-nb-sh", NULL, -1, FALSE },
	{ "webm", "video/webm", NULL, -1, FALSE },
	{ "voc", "audio/x-voc", NULL, -1, FALSE },
	{ "pva", "video/x-pva", NULL, -1, FALSE },
	{ "brstm", "audio/x-brstm", NULL, -1, FALSE },
	{ "bfstm", "audio/x-bfstm", NULL, -1, FALSE },
	{ "avs", "audio/x-bfstm", NULL, -1, FALSE },
	{ "dsf", "audio/x-dsf", NULL, -1, FALSE },
	{ "ea", "video/x-ea", NULL, -1, FALSE },
	{ "film_cpk", "video/x-film-cpk", NULL, -1, FALSE },
	{ "xwma", "audio/x-xwma", NULL, -1, FALSE },
	{ "iff", "application/x-iff", NULL, -1, FALSE },
	{ "idcin", "video/x-idcin", NULL, -1, FALSE },
	{ "ipmovie", "video/x-ipmovie", NULL, -1, FALSE },
	{ "mm", "application/x-mm", NULL, -1, FALSE },
	{ "mmf", "application/vnd.smaf", NULL, -1, FALSE },
	{ "nut", "application/x-nut", NULL, -1, FALSE },
	{ "pxstr", "application/x-pxstr", NULL, -1, FALSE },
	{ "smk", "application/x-smk", NULL, -1, FALSE },
	{ "sol", "application/x-sol", NULL, -1, FALSE },
	{ "vmd", "application/x-vmd", NULL, -1, FALSE },
	{ "wc3movie", "application/x-wc3movie", NULL, -1, FALSE },
	{ "wsaud", "application/x-wsaud", NULL, -1, FALSE },
	{ "wsvqa", "application/x-wsvqa", NULL, -1, FALSE },
};

static struct CodecProps *
find_codec_props (enum AVCodecID id)
{
	int i;

	for (i = 0; i < G_N_ELEMENTS (codec_map); i++) {
		if (codec_map[i].id == id)
			return &codec_map[i];
	}

	return NULL;
}

static struct ContainerProps *
find_container_props (const char *formatid)
{
	int i;

	for (i = 0; i < G_N_ELEMENTS (container_map); i++) {
		if (strcmp (container_map[i].format, formatid) == 0)
			return &container_map[i];
	}

	return NULL;
}

typedef struct _TrackerGUPnPDLNAAudioInformation TrackerGUPnPDLNAAudioInformation;

struct _TrackerGUPnPDLNAAudioInformation {
	GUPnPDLNAAudioInformation parent;
	AVStream *stream;
};

#define TRACKER_TYPE_GUPNP_DLNA_AUDIO_INFORMATION (tracker_gupnp_dlna_audio_information_get_type ())
G_DECLARE_FINAL_TYPE (TrackerGUPnPDLNAAudioInformation,
		      tracker_gupnp_dlna_audio_information,
		      TRACKER, GUPNP_DLNA_AUDIO_INFORMATION,
		      GUPnPDLNAAudioInformation)

G_DEFINE_TYPE (TrackerGUPnPDLNAAudioInformation,
	       tracker_gupnp_dlna_audio_information,
	       GUPNP_TYPE_DLNA_AUDIO_INFORMATION)

#define TO_INT(v) ((GUPnPDLNAIntValue) { .state = GUPNP_DLNA_VALUE_STATE_SET, .value = v })
#define TO_BOOL(v) ((GUPnPDLNABoolValue) { .state = GUPNP_DLNA_VALUE_STATE_SET, .value = v })
#define TO_STRING(v) ((GUPnPDLNAStringValue) { .state = GUPNP_DLNA_VALUE_STATE_SET, .value = g_strdup((v)) })
#define TO_FRAC(n, d) ((GUPnPDLNAFractionValue) { .state = GUPNP_DLNA_VALUE_STATE_SET, .numerator = n, .denominator = d })

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_bitrate (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	return TO_INT (audio_info->stream->codecpar->bit_rate);
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_channels (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	return TO_INT (audio_info->stream->codecpar->ch_layout.nb_channels);
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_depth (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	return TO_INT (audio_info->stream->codecpar->bits_per_coded_sample);
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_layer (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);
	if (prop && prop->layer > 0)
		return TO_INT (prop->layer);
	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_audio_information_get_level (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	g_autofree char *level = NULL;

	if (audio_info->stream->codecpar->level > 0) {
		level = g_strdup_printf ("%d", audio_info->stream->codecpar->level);
		return TO_STRING (level);
	}

	if (audio_info->stream->codecpar->codec_id == AV_CODEC_ID_AAC) {
		if (audio_info->stream->codecpar->ch_layout.nb_channels <= 2 &&
		    audio_info->stream->codecpar->sample_rate <= 24000)
			return TO_STRING ("1");
		else if (audio_info->stream->codecpar->ch_layout.nb_channels <= 2 &&
		         audio_info->stream->codecpar->sample_rate <= 48000)
			return TO_STRING ("2");
		else if (audio_info->stream->codecpar->ch_layout.nb_channels <= 5 &&
		         audio_info->stream->codecpar->sample_rate <= 48000)
			return TO_STRING ("4");
		else if (audio_info->stream->codecpar->ch_layout.nb_channels <= 5 &&
		         audio_info->stream->codecpar->sample_rate <= 96000)
			return TO_STRING ("5");
		else if (audio_info->stream->codecpar->ch_layout.nb_channels <= 7 &&
		         audio_info->stream->codecpar->sample_rate <= 48000)
			return TO_STRING ("6");
		else if (audio_info->stream->codecpar->ch_layout.nb_channels <= 7 &&
		         audio_info->stream->codecpar->sample_rate <= 96000)
			return TO_STRING ("7");
	}

	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_mpeg_audio_version (GUPnPDLNAAudioInformation *info)
{
	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_mpeg_version (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);
	if (prop && prop->mpegversion > 0)
		return TO_INT (prop->mpegversion);

	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_audio_information_get_profile (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);
	if (prop && prop->profile)
		return TO_STRING (prop->profile);

	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_rate (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	return TO_INT (audio_info->stream->codecpar->sample_rate);
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_audio_information_get_stream_format (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);
	if (prop && prop->stream_format)
		return TO_STRING (prop->stream_format);

	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_audio_information_get_wma_version (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);
	if (prop && prop->wmaversion > 0)
		return TO_INT (prop->wmaversion);
	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_audio_information_get_mime (GUPnPDLNAAudioInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (audio_info->stream->codecpar->codec_id);

	if (prop && prop->mime_type)
		return TO_STRING (prop->mime_type);

	return TO_STRING ("");
}

static void
tracker_gupnp_dlna_audio_information_class_init (TrackerGUPnPDLNAAudioInformationClass *klass)
{
	GUPnPDLNAAudioInformationClass *audio_info_class =
		GUPNP_DLNA_AUDIO_INFORMATION_CLASS (klass);

	audio_info_class->get_bitrate =
		tracker_gupnp_dlna_audio_information_get_bitrate;
	audio_info_class->get_channels =
		tracker_gupnp_dlna_audio_information_get_channels;
	audio_info_class->get_depth =
		tracker_gupnp_dlna_audio_information_get_depth;
	audio_info_class->get_layer =
		tracker_gupnp_dlna_audio_information_get_layer;
	audio_info_class->get_level =
		tracker_gupnp_dlna_audio_information_get_level;
	audio_info_class->get_mpeg_audio_version =
		tracker_gupnp_dlna_audio_information_get_mpeg_audio_version;
	audio_info_class->get_mpeg_version =
		tracker_gupnp_dlna_audio_information_get_mpeg_version;
	audio_info_class->get_profile =
		tracker_gupnp_dlna_audio_information_get_profile;
	audio_info_class->get_rate =
		tracker_gupnp_dlna_audio_information_get_rate;
	audio_info_class->get_stream_format =
		tracker_gupnp_dlna_audio_information_get_stream_format;
	audio_info_class->get_wma_version =
		tracker_gupnp_dlna_audio_information_get_wma_version;
	audio_info_class->get_mime =
		tracker_gupnp_dlna_audio_information_get_mime;
}

static void
tracker_gupnp_dlna_audio_information_init (TrackerGUPnPDLNAAudioInformation *info)
{
}

typedef struct _TrackerGUPnPDLNAVideoInformation TrackerGUPnPDLNAVideoInformation;
struct _TrackerGUPnPDLNAVideoInformation {
	GUPnPDLNAVideoInformation parent;
	AVStream *stream;
};

#define TRACKER_TYPE_GUPNP_DLNA_VIDEO_INFORMATION (tracker_gupnp_dlna_video_information_get_type ())
G_DECLARE_FINAL_TYPE (TrackerGUPnPDLNAVideoInformation,
		      tracker_gupnp_dlna_video_information,
		      TRACKER, GUPNP_DLNA_VIDEO_INFORMATION,
		      GUPnPDLNAVideoInformation)

G_DEFINE_TYPE (TrackerGUPnPDLNAVideoInformation,
	       tracker_gupnp_dlna_video_information,
	       GUPNP_TYPE_DLNA_VIDEO_INFORMATION)

static GUPnPDLNAIntValue
tracker_gupnp_dlna_video_information_get_bitrate (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	return TO_INT (video_info->stream->codecpar->bit_rate);
}

static GUPnPDLNAFractionValue
tracker_gupnp_dlna_video_information_get_framerate (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	return TO_FRAC (video_info->stream->codecpar->framerate.num,
			video_info->stream->codecpar->framerate.den);
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_video_information_get_height (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	return TO_INT (video_info->stream->codecpar->height);
}

static GUPnPDLNABoolValue
tracker_gupnp_dlna_video_information_is_interlaced (GUPnPDLNAVideoInformation *info)
{
	return GUPNP_DLNA_BOOL_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_video_information_get_level (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAAudioInformation *audio_info =
		TRACKER_GUPNP_DLNA_AUDIO_INFORMATION (info);
	g_autofree char *level = NULL;

	level = g_strdup_printf ("%d", audio_info->stream->codecpar->level);
	return TO_STRING (level);;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_video_information_get_mpeg_version (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (video_info->stream->codecpar->codec_id);
	if (prop && prop->mpegversion > 0)
		return TO_INT (prop->mpegversion);
	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAFractionValue
tracker_gupnp_dlna_video_information_get_pixel_aspect_ratio (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	return TO_FRAC (video_info->stream->codecpar->sample_aspect_ratio.num,
			video_info->stream->codecpar->sample_aspect_ratio.den);
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_video_information_get_profile (GUPnPDLNAVideoInformation *info)
{
	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNABoolValue
tracker_gupnp_dlna_video_information_is_system_stream (GUPnPDLNAVideoInformation *info)
{
	return GUPNP_DLNA_BOOL_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_video_information_get_width (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	return TO_INT (video_info->stream->codecpar->width);
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_video_information_get_mime (GUPnPDLNAVideoInformation *info)
{
	TrackerGUPnPDLNAVideoInformation *video_info =
		TRACKER_GUPNP_DLNA_VIDEO_INFORMATION (info);
	struct CodecProps *prop;

	prop = find_codec_props (video_info->stream->codecpar->codec_id);

	if (prop && prop->mime_type)
		return TO_STRING (prop->mime_type);

	return TO_STRING ("");
}

static void
tracker_gupnp_dlna_video_information_class_init (TrackerGUPnPDLNAVideoInformationClass *klass)
{
	GUPnPDLNAVideoInformationClass *video_info_class =
		GUPNP_DLNA_VIDEO_INFORMATION_CLASS (klass);

	video_info_class->get_bitrate =
		tracker_gupnp_dlna_video_information_get_bitrate;
	video_info_class->get_framerate =
		tracker_gupnp_dlna_video_information_get_framerate;
	video_info_class->get_height =
		tracker_gupnp_dlna_video_information_get_height;
	video_info_class->is_interlaced =
		tracker_gupnp_dlna_video_information_is_interlaced;
	video_info_class->get_level =
		tracker_gupnp_dlna_video_information_get_level;
	video_info_class->get_mpeg_version =
		tracker_gupnp_dlna_video_information_get_mpeg_version;
	video_info_class->get_pixel_aspect_ratio =
		tracker_gupnp_dlna_video_information_get_pixel_aspect_ratio;
	video_info_class->get_profile =
		tracker_gupnp_dlna_video_information_get_profile;
	video_info_class->is_system_stream =
		tracker_gupnp_dlna_video_information_is_system_stream;
	video_info_class->get_width =
		tracker_gupnp_dlna_video_information_get_width;
	video_info_class->get_mime =
		tracker_gupnp_dlna_video_information_get_mime;
}

static void
tracker_gupnp_dlna_video_information_init (TrackerGUPnPDLNAVideoInformation *info)
{
}

typedef struct _TrackerGUPnPDLNAContainerInformation TrackerGUPnPDLNAContainerInformation;
struct _TrackerGUPnPDLNAContainerInformation {
	GUPnPDLNAContainerInformation parent;
	AVFormatContext *format;
};

#define TRACKER_TYPE_GUPNP_DLNA_CONTAINER_INFORMATION (tracker_gupnp_dlna_container_information_get_type ())
G_DECLARE_FINAL_TYPE (TrackerGUPnPDLNAContainerInformation,
		      tracker_gupnp_dlna_container_information,
		      TRACKER, GUPNP_DLNA_CONTAINER_INFORMATION,
		      GUPnPDLNAContainerInformation)

G_DEFINE_TYPE (TrackerGUPnPDLNAContainerInformation,
	       tracker_gupnp_dlna_container_information,
	       GUPNP_TYPE_DLNA_CONTAINER_INFORMATION)

static GUPnPDLNAIntValue
tracker_gupnp_dlna_container_information_get_mpeg_version (GUPnPDLNAContainerInformation *info)
{
	TrackerGUPnPDLNAContainerInformation *cont_info =
		TRACKER_GUPNP_DLNA_CONTAINER_INFORMATION (info);
	struct ContainerProps *props;

	props = find_container_props (cont_info->format->iformat->name);
	if (props && props->mpegversion > 0)
		return TO_INT (props->mpegversion);

	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAIntValue
tracker_gupnp_dlna_container_information_get_packet_size (GUPnPDLNAContainerInformation *info)
{
	TrackerGUPnPDLNAContainerInformation *cont_info =
		TRACKER_GUPNP_DLNA_CONTAINER_INFORMATION (info);

	if (cont_info->format->packet_size > 0)
		return TO_INT (cont_info->format->packet_size);

	return GUPNP_DLNA_INT_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_container_information_get_profile (GUPnPDLNAContainerInformation *info)
{
	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNABoolValue
tracker_gupnp_dlna_container_information_is_system_stream (GUPnPDLNAContainerInformation *info)
{
	TrackerGUPnPDLNAContainerInformation *cont_info =
		TRACKER_GUPNP_DLNA_CONTAINER_INFORMATION (info);
	struct ContainerProps *props;

	props = find_container_props (cont_info->format->iformat->name);
	if (props)
		return TO_BOOL (props->systemstream);

	return GUPNP_DLNA_BOOL_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_container_information_get_variant (GUPnPDLNAContainerInformation *info)
{
	TrackerGUPnPDLNAContainerInformation *cont_info =
		TRACKER_GUPNP_DLNA_CONTAINER_INFORMATION (info);
	struct ContainerProps *props;

	props = find_container_props (cont_info->format->iformat->name);
	if (props && props->variant)
		return TO_STRING (props->variant);

	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static GUPnPDLNAStringValue
tracker_gupnp_dlna_container_information_get_mime (GUPnPDLNAContainerInformation *info)
{
	TrackerGUPnPDLNAContainerInformation *cont_info =
		TRACKER_GUPNP_DLNA_CONTAINER_INFORMATION (info);
	struct ContainerProps *props;

	props = find_container_props (cont_info->format->iformat->name);
	if (props && props->mime_type)
		return TO_STRING (props->mime_type);

	return GUPNP_DLNA_STRING_VALUE_UNSET;
}

static void
tracker_gupnp_dlna_container_information_class_init (TrackerGUPnPDLNAContainerInformationClass *klass)
{
	GUPnPDLNAContainerInformationClass *cont_info_class =
		GUPNP_DLNA_CONTAINER_INFORMATION_CLASS (klass);

	cont_info_class->get_mpeg_version =
		tracker_gupnp_dlna_container_information_get_mpeg_version;
	cont_info_class->get_packet_size =
		tracker_gupnp_dlna_container_information_get_packet_size;
	cont_info_class->get_profile =
		tracker_gupnp_dlna_container_information_get_profile;
	cont_info_class->is_system_stream =
		tracker_gupnp_dlna_container_information_is_system_stream;
	cont_info_class->get_variant =
		tracker_gupnp_dlna_container_information_get_variant;
	cont_info_class->get_mime =
		tracker_gupnp_dlna_container_information_get_mime;
}

static void
tracker_gupnp_dlna_container_information_init (TrackerGUPnPDLNAContainerInformation *info)
{
}

typedef struct _TrackerGUPnPDLNAInformation TrackerGUPnPDLNAInformation;
struct _TrackerGUPnPDLNAInformation {
	GUPnPDLNAInformation parent;
	AVStream *audio_stream;
	AVStream *video_stream;
	AVFormatContext *format;
};

#define TRACKER_TYPE_GUPNP_DLNA_INFORMATION (tracker_gupnp_dlna_information_get_type ())
G_DECLARE_FINAL_TYPE (TrackerGUPnPDLNAInformation,
		      tracker_gupnp_dlna_information,
		      TRACKER, GUPNP_DLNA_INFORMATION,
		      GUPnPDLNAInformation)

G_DEFINE_TYPE (TrackerGUPnPDLNAInformation,
	       tracker_gupnp_dlna_information,
	       GUPNP_TYPE_DLNA_INFORMATION)

static GUPnPDLNAAudioInformation *
tracker_gupnp_dlna_information_get_audio_information (GUPnPDLNAInformation *info)
{
	TrackerGUPnPDLNAInformation *tracker_info =
		TRACKER_GUPNP_DLNA_INFORMATION (info);
	GUPnPDLNAAudioInformation *audio = NULL;

	if (tracker_info->audio_stream) {
		TrackerGUPnPDLNAAudioInformation *audio_info;

		audio_info = g_object_new (TRACKER_TYPE_GUPNP_DLNA_AUDIO_INFORMATION, NULL);
		audio_info->stream = tracker_info->audio_stream;
		audio = GUPNP_DLNA_AUDIO_INFORMATION (audio_info);
	}

	return audio;
}

static GUPnPDLNAVideoInformation *
tracker_gupnp_dlna_information_get_video_information (GUPnPDLNAInformation *info)
{
	TrackerGUPnPDLNAInformation *tracker_info =
		TRACKER_GUPNP_DLNA_INFORMATION (info);
	GUPnPDLNAVideoInformation *video = NULL;

	if (tracker_info->video_stream) {
		TrackerGUPnPDLNAVideoInformation *video_info;

		video_info = g_object_new (TRACKER_TYPE_GUPNP_DLNA_VIDEO_INFORMATION, NULL);
		video_info->stream = tracker_info->video_stream;
		video = GUPNP_DLNA_VIDEO_INFORMATION (video_info);
	}

	return video;
}

static GUPnPDLNAContainerInformation *
tracker_gupnp_dlna_information_get_container_information (GUPnPDLNAInformation *info)
{
	TrackerGUPnPDLNAInformation *tracker_info =
		TRACKER_GUPNP_DLNA_INFORMATION (info);
	GUPnPDLNAContainerInformation *container = NULL;

	if (tracker_info->audio_stream) {
		TrackerGUPnPDLNAContainerInformation *container_info;

		container_info = g_object_new (TRACKER_TYPE_GUPNP_DLNA_CONTAINER_INFORMATION, NULL);
		container_info->format = tracker_info->format;
		container = GUPNP_DLNA_CONTAINER_INFORMATION (container_info);
	}

	return container;
}

static GUPnPDLNAImageInformation *
tracker_gupnp_dlna_information_get_image_information (GUPnPDLNAInformation *info)
{
	return NULL;
}

static void
tracker_gupnp_dlna_information_class_init (TrackerGUPnPDLNAInformationClass *klass)
{
	GUPnPDLNAInformationClass *dlna_info_class =
		GUPNP_DLNA_INFORMATION_CLASS (klass);

	dlna_info_class->get_audio_information =
		tracker_gupnp_dlna_information_get_audio_information;
	dlna_info_class->get_video_information =
		tracker_gupnp_dlna_information_get_video_information;
	dlna_info_class->get_container_information =
		tracker_gupnp_dlna_information_get_container_information;
	dlna_info_class->get_image_information =
		tracker_gupnp_dlna_information_get_image_information;
}

static void
tracker_gupnp_dlna_information_init (TrackerGUPnPDLNAInformation *info)
{
}

GUPnPDLNAInformation *
tracker_gupnp_dlna_information_new (AVFormatContext *format,
				    AVStream        *audio_stream,
				    AVStream        *video_stream)
{
	TrackerGUPnPDLNAInformation *info;

	info = g_object_new (TRACKER_TYPE_GUPNP_DLNA_INFORMATION, NULL);
	info->audio_stream = audio_stream;
	info->video_stream = video_stream;
	info->format = format;

	return GUPNP_DLNA_INFORMATION (info);
}
