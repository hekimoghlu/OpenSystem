/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
//  SSAlgorithms.h - Description t.b.d.
//
#ifndef _H_SD_ALGORITHMS
#define _H_SD_ALGORITHMS

#include <security_cdsa_plugin/CSPsession.h>

/* Can't include SDCSPDLPlugin.h due to circular dependency */
class SDCSPSession;

// no longer a subclass of AlgorithmFactory due to 
// differing setup() methods
class SDFactory
{
public:
    bool setup(SDCSPSession &session, CSPFullPluginSession::CSPContext * &ctx,
			   const Context &context, bool encoding);
};

#endif // _H_SD_ALGORITHMS
