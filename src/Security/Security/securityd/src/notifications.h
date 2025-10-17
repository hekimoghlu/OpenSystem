/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
// notifications - handling of securityd-gated notification messages
//
#ifndef _H_NOTIFICATIONS
#define _H_NOTIFICATIONS

#include <security_utilities/mach++.h>
#include <security_utilities/machserver.h>
#include <security_utilities/globalizer.h>
#include <securityd_client/ssclient.h>
#include "SharedMemoryCommon.h"
#include <map>
#include <queue>

#include "SharedMemoryServer.h"

using MachPlusPlus::Port;
using MachPlusPlus::MachServer;
using SecurityServer::NotificationDomain;
using SecurityServer::NotificationEvent;
using SecurityServer::NotificationMask;

class SharedMemoryListener;

//
// A registered receiver of notifications.
// This is an abstract class; you must subclass to define notifyMe().
//
// All Listeners in existence are collected in an internal map of ports to
// Listener*s, which makes them eligible to have events delivered to them via
// their notifyMe() method. There are (only) two viable lifetime management
// strategies for your Listener subclass:
// (1) Eternal: don't ever destroy your Listener. All is well. By convention,
// such Listeners use the null port.
// (2) Port-based: To get rid of your Listeners, call Listener::remove(port),
// which will delete(!) all Listeners constructed with that port.
// Except for the remove() functionality, Listener does not interpret the port.
//
// If you need another Listener lifetime management strategy, you will probably
// have to change things around here.
//
class Listener: public RefCount {
public:
	Listener(NotificationDomain domain, NotificationMask events,
		mach_port_t port = MACH_PORT_NULL);	
	virtual ~Listener();

	// inject an event into the notification system
    static void notify(NotificationDomain domain,
		NotificationEvent event, const CssmData &data);
    static void notify(NotificationDomain domain,
		NotificationEvent event, uint32 sequence, const CssmData &data, audit_token_t auditToken);

    const NotificationDomain domain;
    const NotificationMask events;
	
	bool wants(NotificationEvent event)
	{ return (1 << event) & events; }

protected:
	class Notification : public RefCount {
	public:
		Notification(NotificationDomain domain, NotificationEvent event,
			uint32 seq, const CssmData &data);
		virtual ~Notification();
		
		const NotificationDomain domain;
		const NotificationEvent event;
		const uint32 sequence;
		const CssmAutoData data;

        std::string description() const;
		size_t size() const
		{ return data.length(); }	//@@@ add "slop" here for heuristic?
	};
	
	virtual void notifyMe(Notification *message) = 0;

    static bool testPredicate(const std::function<bool(const Listener& listener)> test);

public:
	class JitterBuffer {
	public:
		JitterBuffer() : mNotifyLast(0) { }

		bool inSequence(Notification *message);
		RefPointer<Notification> popNotification();
		
	private:
		uint32 mNotifyLast;		// last notification seq processed
		typedef std::map<uint32, RefPointer<Notification> > JBuffer;
		JBuffer mBuffer;		// early messages buffer
	};
	
private:
	static void sendNotification(Notification *message);
    
private:
    typedef multimap<mach_port_t, RefPointer<Listener> > ListenerMap;
    static ListenerMap& listeners;
    static Mutex setLock;
};



class SharedMemoryListener : public Listener, public SharedMemoryServer, public Security::MachPlusPlus::MachServer::Timer
{
protected:
	virtual void action ();
	virtual void notifyMe(Notification *message);

    static bool findUID(uid_t uid);
    static int get_process_euid(pid_t pid, uid_t& out_euid);

    bool needsPrivacyFilter(Notification *notification);
    bool isTrustEvent(Notification *notification);
    uint32 getRecordType(const CssmData& val) const;
    
	bool mActive;

    Mutex mMutex;

public:
	SharedMemoryListener (const char* serverName, u_int32_t serverSize, uid_t uid = 0, gid_t gid = 0);
	virtual ~SharedMemoryListener ();

    static void createDefaultSharedMemoryListener(uid_t uid, gid_t gid);
};

#endif
