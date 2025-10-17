/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#ifndef _UAPI_AMT_H_
#define _UAPI_AMT_H_
enum ifla_amt_mode {
  AMT_MODE_GATEWAY = 0,
  AMT_MODE_RELAY,
  __AMT_MODE_MAX,
};
#define AMT_MODE_MAX (__AMT_MODE_MAX - 1)
enum {
  IFLA_AMT_UNSPEC,
  IFLA_AMT_MODE,
  IFLA_AMT_RELAY_PORT,
  IFLA_AMT_GATEWAY_PORT,
  IFLA_AMT_LINK,
  IFLA_AMT_LOCAL_IP,
  IFLA_AMT_REMOTE_IP,
  IFLA_AMT_DISCOVERY_IP,
  IFLA_AMT_MAX_TUNNELS,
  __IFLA_AMT_MAX,
};
#define IFLA_AMT_MAX (__IFLA_AMT_MAX - 1)
#endif
