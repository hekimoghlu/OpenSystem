/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef _UAPI__KERNELCAPI_H__
#define _UAPI__KERNELCAPI_H__
#define CAPI_MAXAPPL 240
#define CAPI_MAXCONTR 32
#define CAPI_MAXDATAWINDOW 8
typedef struct kcapi_flagdef {
  int contr;
  int flag;
} kcapi_flagdef;
typedef struct kcapi_carddef {
  char driver[32];
  unsigned int port;
  unsigned irq;
  unsigned int membase;
  int cardnr;
} kcapi_carddef;
#define KCAPI_CMD_TRACE 10
#define KCAPI_CMD_ADDCARD 11
#define KCAPI_TRACE_OFF 0
#define KCAPI_TRACE_SHORT_NO_DATA 1
#define KCAPI_TRACE_FULL_NO_DATA 2
#define KCAPI_TRACE_SHORT 3
#define KCAPI_TRACE_FULL 4
#endif
