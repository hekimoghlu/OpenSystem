/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
//
// dlclientpriv - private client interface to CSSM DLs
//
// This file implements those (non-virtual) methods of Db/DbImpl that
// require additional libraries to function. The OS X linker is too inept
// to eliminate unused functions peacefully (as of OS X 10.3/XCode 1.5 anyway).
//
#include <security_cdsa_client/dlclient.h>
#include <security_cdsa_client/aclclient.h>
#include <securityd_client/ssclient.h>

using namespace CssmClient;


//
// Currently empty.
//
