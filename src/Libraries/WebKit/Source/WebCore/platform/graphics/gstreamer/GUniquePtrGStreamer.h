/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#if USE(GSTREAMER)

#include <gst/audio/audio.h>
#include <gst/base/gstbytereader.h>
#include <gst/base/gstflowcombiner.h>
#include <gst/fft/gstfftf32.h>
#include <gst/gstsegment.h>
#include <gst/gststructure.h>
#include <gst/pbutils/install-plugins.h>
#include <gst/video/video.h>
#include <wtf/glib/GUniquePtr.h>

#if defined(BUILDING_WebCore) && USE(GSTREAMER_WEBRTC)
#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
#undef GST_USE_UNSTABLE_API
#endif

namespace WTF {

WTF_DEFINE_GPTR_DELETER(GstStructure, gst_structure_free)
WTF_DEFINE_GPTR_DELETER(GstInstallPluginsContext, gst_install_plugins_context_free)
WTF_DEFINE_GPTR_DELETER(GstIterator, gst_iterator_free)
WTF_DEFINE_GPTR_DELETER(GstSegment, gst_segment_free)
WTF_DEFINE_GPTR_DELETER(GstFlowCombiner, gst_flow_combiner_free)
WTF_DEFINE_GPTR_DELETER(GstByteReader, gst_byte_reader_free)
WTF_DEFINE_GPTR_DELETER(GstVideoConverter, gst_video_converter_free)
WTF_DEFINE_GPTR_DELETER(GstAudioConverter, gst_audio_converter_free)
WTF_DEFINE_GPTR_DELETER(GstAudioInfo, gst_audio_info_free)
WTF_DEFINE_GPTR_DELETER(GstFFTF32, gst_fft_f32_free)

#if defined(BUILDING_WebCore) && USE(GSTREAMER_WEBRTC)
WTF_DEFINE_GPTR_DELETER(GstWebRTCSessionDescription, gst_webrtc_session_description_free)
WTF_DEFINE_GPTR_DELETER(GstSDPMessage, gst_sdp_message_free)
#endif
}

#endif // USE(GSTREAMER)

