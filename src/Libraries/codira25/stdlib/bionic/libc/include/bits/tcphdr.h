/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include <sys/cdefs.h>
#include <stdint.h>

__BEGIN_DECLS

struct tcphdr {
  __extension__ union {
    struct {
      uint16_t th_sport;
      uint16_t th_dport;
      uint32_t th_seq;
      uint32_t th_ack;
      uint8_t th_x2:4;
      uint8_t th_off:4;
      uint8_t th_flags;
      uint16_t th_win;
      uint16_t th_sum;
      uint16_t th_urp;
    };
    struct {
      uint16_t source;
      uint16_t dest;
      uint32_t seq;
      uint32_t ack_seq;
      uint16_t res1:4;
      uint16_t doff:4;
      uint16_t fin:1;
      uint16_t syn:1;
      uint16_t rst:1;
      uint16_t psh:1;
      uint16_t ack:1;
      uint16_t urg:1;
      uint16_t res2:2;
      uint16_t window;
      uint16_t check;
      uint16_t urg_ptr;
    };
  };
};

__END_DECLS
