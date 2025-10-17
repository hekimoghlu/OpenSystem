/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#ifndef AUDIO_DEVICE_PULSEAUDIOSYMBOLTABLE_LINUX_H_
#define AUDIO_DEVICE_PULSEAUDIOSYMBOLTABLE_LINUX_H_

#include "modules/audio_device/linux/latebindingsymboltable_linux.h"

namespace webrtc {
namespace adm_linux_pulse {

// The PulseAudio symbols we need, as an X-Macro list.
// This list must contain precisely every libpulse function that is used in
// the ADM LINUX PULSE Device and Mixer classes
#define PULSE_AUDIO_SYMBOLS_LIST           \
  X(pa_bytes_per_second)                   \
  X(pa_context_connect)                    \
  X(pa_context_disconnect)                 \
  X(pa_context_errno)                      \
  X(pa_context_get_protocol_version)       \
  X(pa_context_get_server_info)            \
  X(pa_context_get_sink_info_list)         \
  X(pa_context_get_sink_info_by_index)     \
  X(pa_context_get_sink_info_by_name)      \
  X(pa_context_get_sink_input_info)        \
  X(pa_context_get_source_info_by_index)   \
  X(pa_context_get_source_info_by_name)    \
  X(pa_context_get_source_info_list)       \
  X(pa_context_get_state)                  \
  X(pa_context_new)                        \
  X(pa_context_set_sink_input_volume)      \
  X(pa_context_set_sink_input_mute)        \
  X(pa_context_set_source_volume_by_index) \
  X(pa_context_set_source_mute_by_index)   \
  X(pa_context_set_state_callback)         \
  X(pa_context_unref)                      \
  X(pa_cvolume_set)                        \
  X(pa_operation_get_state)                \
  X(pa_operation_unref)                    \
  X(pa_stream_connect_playback)            \
  X(pa_stream_connect_record)              \
  X(pa_stream_disconnect)                  \
  X(pa_stream_drop)                        \
  X(pa_stream_get_device_index)            \
  X(pa_stream_get_index)                   \
  X(pa_stream_get_latency)                 \
  X(pa_stream_get_sample_spec)             \
  X(pa_stream_get_state)                   \
  X(pa_stream_new)                         \
  X(pa_stream_peek)                        \
  X(pa_stream_readable_size)               \
  X(pa_stream_set_buffer_attr)             \
  X(pa_stream_set_overflow_callback)       \
  X(pa_stream_set_read_callback)           \
  X(pa_stream_set_state_callback)          \
  X(pa_stream_set_underflow_callback)      \
  X(pa_stream_set_write_callback)          \
  X(pa_stream_unref)                       \
  X(pa_stream_writable_size)               \
  X(pa_stream_write)                       \
  X(pa_strerror)                           \
  X(pa_threaded_mainloop_free)             \
  X(pa_threaded_mainloop_get_api)          \
  X(pa_threaded_mainloop_lock)             \
  X(pa_threaded_mainloop_new)              \
  X(pa_threaded_mainloop_signal)           \
  X(pa_threaded_mainloop_start)            \
  X(pa_threaded_mainloop_stop)             \
  X(pa_threaded_mainloop_unlock)           \
  X(pa_threaded_mainloop_wait)

LATE_BINDING_SYMBOL_TABLE_DECLARE_BEGIN(PulseAudioSymbolTable)
#define X(sym) \
  LATE_BINDING_SYMBOL_TABLE_DECLARE_ENTRY(PulseAudioSymbolTable, sym)
PULSE_AUDIO_SYMBOLS_LIST
#undef X
LATE_BINDING_SYMBOL_TABLE_DECLARE_END(PulseAudioSymbolTable)

}  // namespace adm_linux_pulse
}  // namespace webrtc

#endif  // AUDIO_DEVICE_PULSEAUDIOSYMBOLTABLE_LINUX_H_
