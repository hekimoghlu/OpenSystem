/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef _UAPI_IIO_EVENTS_H_
#define _UAPI_IIO_EVENTS_H_
#include <linux/ioctl.h>
#include <linux/types.h>
struct iio_event_data {
  __u64 id;
  __s64 timestamp;
};
#define IIO_GET_EVENT_FD_IOCTL _IOR('i', 0x90, int)
#define IIO_EVENT_CODE_EXTRACT_TYPE(mask) ((mask >> 56) & 0xFF)
#define IIO_EVENT_CODE_EXTRACT_DIR(mask) ((mask >> 48) & 0x7F)
#define IIO_EVENT_CODE_EXTRACT_CHAN_TYPE(mask) ((mask >> 32) & 0xFF)
#define IIO_EVENT_CODE_EXTRACT_CHAN(mask) ((__s16) (mask & 0xFFFF))
#define IIO_EVENT_CODE_EXTRACT_CHAN2(mask) ((__s16) (((mask) >> 16) & 0xFFFF))
#define IIO_EVENT_CODE_EXTRACT_MODIFIER(mask) ((mask >> 40) & 0xFF)
#define IIO_EVENT_CODE_EXTRACT_DIFF(mask) (((mask) >> 55) & 0x1)
#endif
