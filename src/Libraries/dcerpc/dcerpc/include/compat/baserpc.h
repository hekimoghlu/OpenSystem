/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#ifndef DCERPC_H
#define DCERPC_H
#if !defined(USE_DCE_STYLE) && !defined(USE_MS_STYLE)
#error Define either USE_DCE_STYLE or USE_MS_STYLE to decide which rpc call style should be used.
#endif

#ifndef _WIN32
#ifdef USE_DCE_STYLE
#include <dce/rpc.h>
#include <dce/pthread_exc.h>
#include <dce/dce_error.h>
#else
#include "mswrappers.h"
#endif
#else //_WIN32 defined
#include <rpc.h>
#ifdef USE_DCE_STYLE
#include "dce2msrpc.h"
#endif
#endif //_WIN32

#include "rpcfields.h"
#endif //DCERPC_H
