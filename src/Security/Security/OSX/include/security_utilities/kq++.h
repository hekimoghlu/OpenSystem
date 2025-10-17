/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#ifndef _H_KQPP
#define _H_KQPP

#include <security_utilities/unix++.h>
#include <sys/event.h>

namespace Security {
namespace UnixPlusPlus {


class KEvent;


class KQueue {
public:
	KQueue();
	virtual ~KQueue();

	unsigned operator () (const KEvent *updates, unsigned updateCount, KEvent *events, unsigned eventCount,
		const timespec *timeout = NULL);
	unsigned operator () (KEvent *events, unsigned eventCount, const timespec *timeout = NULL)
		{ return operator () (NULL, NULL, events, eventCount, timeout); }
	
	void update(const KEvent &event, unsigned flags = EV_ADD);
	bool receive(KEvent &event, const timespec *timeout = NULL);

private:
	int mQueue;
};


class KEvent : public PodWrapper<KEvent, kevent64_s> {
public:
	KEvent() { clearPod(); }
	KEvent(int16_t filt) { clearPod(); this->filter = filt; }
	KEvent(int16_t filt, uint64_t id, uint32_t ffl = 0)
//		{ clearPod(); this->ident = id; this->filter = filt; this->fflags = ffl; }
		{ EV_SET64(this, id, filt, 0, ffl, 0, 0, 0, 0); }

	void addTo(KQueue &kq, unsigned flags = 0)
		{ this->flags = EV_ADD | flags; kq.update(*this); }
	void removeFrom(KQueue &kq, unsigned flags = 0)
		{ this->flags = EV_DELETE | flags; kq.update(*this); }
	void enable(KQueue &kq, unsigned flags = 0)
		{ this->flags = EV_ENABLE | flags; kq.update(*this); }
	void disable(KQueue &kq, unsigned flags = 0)
		{ this->flags = EV_DISABLE | flags; kq.update(*this); }
};


namespace Event {


class Vnode : public KEvent {
public:
	Vnode() : KEvent(EVFILT_VNODE) { }
	Vnode(int fd, uint32_t flags) : KEvent(EVFILT_VNODE, fd, flags) { }
	
	int fd() const { return (int)this->ident; }
};

} // namespace Event


} // end namespace UnixPlusPlus
} // end namespace Security

#endif //_H_KQPP
