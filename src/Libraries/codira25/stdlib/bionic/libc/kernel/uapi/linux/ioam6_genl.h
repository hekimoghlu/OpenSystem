/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#ifndef _UAPI_LINUX_IOAM6_GENL_H
#define _UAPI_LINUX_IOAM6_GENL_H
#define IOAM6_GENL_NAME "IOAM6"
#define IOAM6_GENL_VERSION 0x1
enum {
  IOAM6_ATTR_UNSPEC,
  IOAM6_ATTR_NS_ID,
  IOAM6_ATTR_NS_DATA,
  IOAM6_ATTR_NS_DATA_WIDE,
#define IOAM6_MAX_SCHEMA_DATA_LEN (255 * 4)
  IOAM6_ATTR_SC_ID,
  IOAM6_ATTR_SC_DATA,
  IOAM6_ATTR_SC_NONE,
  IOAM6_ATTR_PAD,
  __IOAM6_ATTR_MAX,
};
#define IOAM6_ATTR_MAX (__IOAM6_ATTR_MAX - 1)
enum {
  IOAM6_CMD_UNSPEC,
  IOAM6_CMD_ADD_NAMESPACE,
  IOAM6_CMD_DEL_NAMESPACE,
  IOAM6_CMD_DUMP_NAMESPACES,
  IOAM6_CMD_ADD_SCHEMA,
  IOAM6_CMD_DEL_SCHEMA,
  IOAM6_CMD_DUMP_SCHEMAS,
  IOAM6_CMD_NS_SET_SCHEMA,
  __IOAM6_CMD_MAX,
};
#define IOAM6_CMD_MAX (__IOAM6_CMD_MAX - 1)
#define IOAM6_GENL_EV_GRP_NAME "ioam6_events"
enum ioam6_event_type {
  IOAM6_EVENT_UNSPEC,
  IOAM6_EVENT_TRACE,
};
enum ioam6_event_attr {
  IOAM6_EVENT_ATTR_UNSPEC,
  IOAM6_EVENT_ATTR_TRACE_NAMESPACE,
  IOAM6_EVENT_ATTR_TRACE_NODELEN,
  IOAM6_EVENT_ATTR_TRACE_TYPE,
  IOAM6_EVENT_ATTR_TRACE_DATA,
  __IOAM6_EVENT_ATTR_MAX
};
#define IOAM6_EVENT_ATTR_MAX (__IOAM6_EVENT_ATTR_MAX - 1)
#endif
