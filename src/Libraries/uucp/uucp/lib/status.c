/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#include "uucp.h"

#include "uudefs.h"

/* Status strings.  These must match enum tstatus_type.  */

#if USE_TRADITIONAL_STATUS

const char *azStatus[] =
{
  "SUCCESSFUL",
  "DEVICE FAILED",
  "DIAL FAILED",
  "LOGIN FAILED",
  "STARTUP FAILED",
  "CONVERSATION FAILED",
  "TALKING",
  "WRONG TIME TO CALL"
};

#else

const char *azStatus[] =
{
  "Conversation complete",
  "Port unavailable",
  "Dial failed",
  "Login failed",
  "Handshake failed",
  "Call failed",
  "Talking",
  "Wrong time to call"
};

#endif
