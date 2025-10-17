/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
// DigestContext.cpp 
//
#include "DigestContext.h"
#include <AppleCSPUtils.h>

/* 
 * Just field the expected/required calls from CSPFullPluginSession,
 * and dispatch them to mDigest.
 */
void DigestContext::init(const Context &context, bool)
{
	mDigest.digestInit();
}

void DigestContext::update(const CssmData &data)
{
	mDigest.digestUpdate((const uint8 *)data.data(), data.length());
}

void DigestContext::final(CssmData &data)
{
	data.length(mDigest.digestSizeInBytes());
	mDigest.digestFinal((uint8 *)data.data());
}

CSPFullPluginSession::CSPContext *DigestContext::clone(Allocator &)
{
	/* first clone the low-level digest object */
	DigestObject *newDigest = mDigest.digestClone();
	
	/* now construct a new context */
	return new DigestContext(session(), *newDigest);
}

size_t DigestContext::outputSize(bool, size_t) 
{
	return mDigest.digestSizeInBytes();
}

