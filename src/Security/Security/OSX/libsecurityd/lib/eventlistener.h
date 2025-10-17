/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef _H_EVENTLISTENER
#define _H_EVENTLISTENER

#include <securityd_client/ssclient.h>
#include <security_utilities/cfmach++.h>
#include <security_utilities/refcount.h>

namespace Security {
namespace SecurityServer {


//
// A CFNotificationDispatcher registers with the local CFRunLoop to automatically
// receive notification messages and dispatch them.
//
class EventListener : public RefCount
{
protected:
    bool mInitialized;
	NotificationDomain mDomain;
	NotificationMask mMask;

    static dispatch_once_t queueOnceToken;
    static dispatch_queue_t notificationQueue;
public:
    static dispatch_queue_t getNotificationQueue();

	EventListener(NotificationDomain domain, NotificationMask eventMask);
	virtual ~EventListener();

    virtual bool initialized()  { return mInitialized; }

	virtual void consume(NotificationDomain domain, NotificationEvent event, const CssmData& data);
	
	NotificationDomain GetDomain () {return mDomain;}
	NotificationMask GetMask () {return mMask;}
    
    static void FinishedInitialization(EventListener* eventListener);
};

// For backward compatiblity, we remember the client's CFRunLoop when notifications are enabled.
// Use this function to get this run loop, to route notifications back to them on it.
CFRunLoopRef clientNotificationRunLoop();


} // end namespace SecurityServer
} // end namespace Security


#endif
