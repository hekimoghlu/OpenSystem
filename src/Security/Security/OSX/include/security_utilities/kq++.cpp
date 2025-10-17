/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
// kq++ - kqueue/kevent interface
//
#include <security_utilities/kq++.h>

namespace Security {
namespace UnixPlusPlus {


KQueue::KQueue()
{
	UnixError::check(mQueue = ::kqueue());
}

KQueue::~KQueue()
{
	UnixError::check(::close(mQueue));
}


unsigned KQueue::operator () (const KEvent *updates, unsigned updateCount,
	KEvent *events, unsigned eventCount, const timespec *timeout)
{
	int rc = ::kevent64(mQueue, updates, updateCount, events, eventCount, 0, timeout);
	UnixError::check(rc);
	assert(rc >= 0);
	return rc;
}


void KQueue::update(const KEvent &event, unsigned flags)
{
	KEvent ev = event;
	ev.flags = flags;
	(*this)(&event, 1, NULL, NULL);
}

bool KQueue::receive(KEvent &event, const timespec *timeout)
{
	return (*this)(&event, 1, timeout) > 0;
}


} // end namespace UnixPlusPlus
} // end namespace Security
