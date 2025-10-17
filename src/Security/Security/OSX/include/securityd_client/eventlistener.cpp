/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#include <list>
#include <security_utilities/globalizer.h>
#include <security_utilities/threading.h>
#include "eventlistener.h"
#include "SharedMemoryClient.h"
#include <notify.h>
#include "sscommon.h"
#include <sys/syslog.h>
#include <Security/SecBasePriv.h>

using namespace MachPlusPlus;


namespace Security {
namespace SecurityServer {

typedef RefPointer<EventListener> EventPointer;
typedef std::list<EventPointer> EventListenerList;

static const char* GetNotificationName ()
{
	return SharedMemoryCommon::kDefaultSecurityMessagesName;
}



class SharedMemoryClientMaker
{
private:
	SharedMemoryClient mClient;

public:
	SharedMemoryClientMaker ();
	SharedMemoryClient* Client ();
};



SharedMemoryClientMaker::SharedMemoryClientMaker () : mClient (GetNotificationName (), kSharedMemoryPoolSize, SharedMemoryCommon::fixUID(getuid()))
{
    secdebug("MDSPRIVACY","[%03d] SharedMemoryClientMaker uid: %d, euid: %d, name: %s", mClient.getUID(), getuid(), geteuid(), GetNotificationName ());
}



SharedMemoryClient* SharedMemoryClientMaker::Client ()
{
	return &mClient;
}



ModuleNexus<EventListenerList> gEventListeners;
ModuleNexus<Mutex> gNotificationLock;
ModuleNexus<SharedMemoryClientMaker> gMemoryClient;

//
// Note that once we start notifications, we want receive them forever. Don't have a cancel option.
//
static bool InitializeNotifications () {
    bool initializationComplete = false;
    static dispatch_queue_t notification_queue = EventListener::getNotificationQueue();

    secdebug("MDSPRIVACY","EventListener Init: uid: %d, euid: %d, name: %s", getuid(), geteuid(), GetNotificationName ());
    // Initialize the memory client
    gMemoryClient();

    if (gMemoryClient().Client()->uninitialized()) {
        secdebug("MDSPRIVACY","[%03d] FATAL: InitializeNotifications EventListener uninitialized; process will never get keychain notifications", getuid());
        return initializationComplete;
    }
    int out_token;

    notify_handler_t receive = ^(int token){
        try {
            SegmentOffsetType length;
            UnavailableReason ur;

            bool result;

            // Trust the memory client to break our loop here
            while (true)
            {
                u_int8_t *buffer = new u_int8_t[kSharedMemoryPoolSize];
                {
                    StLock<Mutex> lock (gNotificationLock ());
                    result = gMemoryClient().Client()->ReadMessage(buffer, length, ur);
                    if (!result)
                    {
                        secdebug("MDSPRIVACY","[%03d] notify_handler ReadMessage ur: %d", getuid(), ur);
                        delete [] buffer;
                        return;
                    }
                }

                // Send this event off to the listeners
                {
                    StLock<Mutex> lock (gNotificationLock ());
                    EventListenerList& eventList = gEventListeners();
                    std::set<EventPointer *> processedListeners;

                    // route the message to its destination
                    u_int32_t* ptr = (u_int32_t*) buffer;

                    // we have a message, do the semantics...
                    SecurityServer::NotificationDomain domain = (SecurityServer::NotificationDomain) OSSwapBigToHostInt32 (*ptr++);
                    SecurityServer::NotificationEvent event = (SecurityServer::NotificationEvent) OSSwapBigToHostInt32 (*ptr++);
                    CssmData data ((u_int8_t*) ptr, buffer + length - (u_int8_t*) ptr);

                    string descrip = SharedMemoryCommon::notificationDescription(domain, event);
                    secdebug("MDSPRIVACY","[%03d] notify_handler: %s", getuid(), descrip.c_str());
                    EventListenerList::iterator it = eventList.begin ();
                    while (it != eventList.end ())
                    {
                        EventPointer ep = *it++;
                        /*
                         * We can't hold the global event lock when processing callback handers
                         * so remember what items we have processes and don't process them again.
                         * and when we have done a callback we loop back and try again.
                         */
                        if (processedListeners.find(&ep) != processedListeners.end()) {
                            continue;
                        }
                        processedListeners.insert(&ep);

                        /* is this event for this Event handler */
                        if (ep->GetDomain() != domain || (ep->GetMask() & (1 << event)) == 0)
                            continue;

                        lock.unlock();

                        try {
                            ep->consume (domain, event, data);
                        } catch (CssmError &e) {
                            // If we throw, libnotify will abort the process. Log these...
                            secerror("caught CssmError while processing notification: %d %s", e.error, cssmErrorString(e.error));
                        }
                        lock.lock();
                        /*
                         * If we have to grab the lock again, start over iteration
                         */
                        it = eventList.begin ();
                    }
                }

                delete [] buffer;
            }
        }
        // If these exceptions propagate, we crash our enclosing app. That's bad. Worse than silently swallowing the error.
        catch(CssmError &cssme) {
            secerror("caught CssmError during notification: %d %s", (int) cssme.error, cssmErrorString(cssme.error));
        }
        catch(UnixError &ue) {
            secerror("caught UnixError during notification: %d %s", ue.unixError(), ue.what());
        }
        catch (MacOSError mose) {
            secerror("caught MacOSError during notification: %d %s", (int) mose.osStatus(), mose.what());
        }
        catch (...) {
            secerror("caught unknown error during notification");
        }
    };

    uint32_t status = notify_register_dispatch(GetNotificationName(), &out_token, notification_queue, receive);
    if (status) {
        secerror("notify_register_dispatch failed: %d", status);
        syslog(LOG_ERR, "notify_register_dispatch failed: %d", status);
    } else {
        initializationComplete = true;
    }
    return initializationComplete;
}


EventListener::EventListener (NotificationDomain domain, NotificationMask eventMask)
	: mInitialized(false), mDomain (domain), mMask (eventMask)
{
	// make sure that notifications are turned on.
	mInitialized = InitializeNotifications();
}

//
// StopNotification() is needed on destruction; everyone else cleans up after themselves.
//
EventListener::~EventListener () {
    if (initialized()) {
        StLock<Mutex> lock (gNotificationLock ());

        // find the listener in the list and remove it
        EventListenerList::iterator it = std::find (gEventListeners ().begin (),
                                                    gEventListeners ().end (),
                                                    this);
        if (it != gEventListeners ().end ())
        {
            gEventListeners ().erase (it);
        }
    }
}

// get rid of the pure virtual
void EventListener::consume(NotificationDomain, NotificationEvent, const Security::CssmData&)
{
}



void EventListener::FinishedInitialization(EventListener *eventListener) {
    if (eventListener->initialized()) {
        StLock<Mutex> lock (gNotificationLock ());
        gEventListeners().push_back (eventListener);
    }
}

dispatch_once_t EventListener::queueOnceToken = 0;
dispatch_queue_t EventListener::notificationQueue = NULL;

dispatch_queue_t EventListener::getNotificationQueue() {
    dispatch_once(&queueOnceToken, ^{
        notificationQueue = dispatch_queue_create("com.apple.security.keychain-notification-queue", DISPATCH_QUEUE_SERIAL);
    });

    return notificationQueue;
}


} // end namespace SecurityServer
} // end namespace Security
