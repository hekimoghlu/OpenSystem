/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#ifndef _UAPI_LINUX_IOAM6_IPTUNNEL_H
#define _UAPI_LINUX_IOAM6_IPTUNNEL_H
enum {
  __IOAM6_IPTUNNEL_MODE_MIN,
  IOAM6_IPTUNNEL_MODE_INLINE,
  IOAM6_IPTUNNEL_MODE_ENCAP,
  IOAM6_IPTUNNEL_MODE_AUTO,
  __IOAM6_IPTUNNEL_MODE_MAX,
};
#define IOAM6_IPTUNNEL_MODE_MIN (__IOAM6_IPTUNNEL_MODE_MIN + 1)
#define IOAM6_IPTUNNEL_MODE_MAX (__IOAM6_IPTUNNEL_MODE_MAX - 1)
enum {
  IOAM6_IPTUNNEL_UNSPEC,
  IOAM6_IPTUNNEL_MODE,
  IOAM6_IPTUNNEL_DST,
  IOAM6_IPTUNNEL_TRACE,
#define IOAM6_IPTUNNEL_FREQ_MIN 1
#define IOAM6_IPTUNNEL_FREQ_MAX 1000000
  IOAM6_IPTUNNEL_FREQ_K,
  IOAM6_IPTUNNEL_FREQ_N,
  IOAM6_IPTUNNEL_SRC,
  __IOAM6_IPTUNNEL_MAX,
};
#define IOAM6_IPTUNNEL_MAX (__IOAM6_IPTUNNEL_MAX - 1)
#endif
