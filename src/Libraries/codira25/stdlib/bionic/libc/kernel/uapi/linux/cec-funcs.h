/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#ifndef _CEC_UAPI_FUNCS_H
#define _CEC_UAPI_FUNCS_H
#include <linux/cec.h>
struct cec_op_arib_data {
  __u16 transport_id;
  __u16 service_id;
  __u16 orig_network_id;
};
struct cec_op_atsc_data {
  __u16 transport_id;
  __u16 program_number;
};
struct cec_op_dvb_data {
  __u16 transport_id;
  __u16 service_id;
  __u16 orig_network_id;
};
struct cec_op_channel_data {
  __u8 channel_number_fmt;
  __u16 major;
  __u16 minor;
};
struct cec_op_digital_service_id {
  __u8 service_id_method;
  __u8 dig_bcast_system;
  union {
    struct cec_op_arib_data arib;
    struct cec_op_atsc_data atsc;
    struct cec_op_dvb_data dvb;
    struct cec_op_channel_data channel;
  };
};
struct cec_op_record_src {
  __u8 type;
  union {
    struct cec_op_digital_service_id digital;
    struct {
      __u8 ana_bcast_type;
      __u16 ana_freq;
      __u8 bcast_system;
    } analog;
    struct {
      __u8 plug;
    } ext_plug;
    struct {
      __u16 phys_addr;
    } ext_phys_addr;
  };
};
struct cec_op_tuner_device_info {
  __u8 rec_flag;
  __u8 tuner_display_info;
  __u8 is_analog;
  union {
    struct cec_op_digital_service_id digital;
    struct {
      __u8 ana_bcast_type;
      __u16 ana_freq;
      __u8 bcast_system;
    } analog;
  };
};
struct cec_op_ui_command {
  __u8 ui_cmd;
  __u8 has_opt_arg;
  union {
    struct cec_op_channel_data channel_identifier;
    __u8 ui_broadcast_type;
    __u8 ui_sound_presentation_control;
    __u8 play_mode;
    __u8 ui_function_media;
    __u8 ui_function_select_av_input;
    __u8 ui_function_select_audio_input;
  };
};
#endif
