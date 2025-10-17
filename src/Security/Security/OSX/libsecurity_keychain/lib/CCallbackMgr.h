/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
/*
 *  CCallbackMgr.h -- Code that communicates with processes that install a callback
 *  with the Keychain Manager to receive keychain events.
 */
#ifndef _SECURITY_CCALLBACKMGR_H_
#define _SECURITY_CCALLBACKMGR_H_

#include <security_keychain/Keychains.h>
#include <security_utilities/cfmach++.h>
#include <securityd_client/ssnotify.h>
#include <securityd_client/dictionary.h>
#include <securityd_client/eventlistener.h>
#include <list>
#include "KCEventNotifier.h"

namespace Security
{

namespace KeychainCore
{

class CallbackInfo;
class CCallbackMgr;

class CallbackInfo
{
public:
	~CallbackInfo();
	CallbackInfo();
	CallbackInfo(SecKeychainCallback inCallbackFunction,SecKeychainEventMask inEventMask,void *inContext, CFRunLoopRef runLoop);
    CallbackInfo(const CallbackInfo& cb);
	
	bool operator ==(const CallbackInfo& other) const;
	bool operator !=(const CallbackInfo& other) const;

	SecKeychainCallback mCallback;
	SecKeychainEventMask mEventMask;
	void *mContext;
    CFRunLoopRef mRunLoop;
    bool mActive;
};

// typedefs
typedef CallbackInfo *CallbackInfoPtr;
typedef CallbackInfo const *ConstCallbackInfoPtr;

typedef list<CallbackInfo>::iterator CallbackInfoListIterator;
typedef list<CallbackInfo>::const_iterator ConstCallbackInfoListIterator;


class CCallbackMgr : public SecurityServer::EventListener
{
public:
	CCallbackMgr();
	~CCallbackMgr();
	
	static CCallbackMgr& Instance();

	static void AddCallback( SecKeychainCallback inCallbackFunction, SecKeychainEventMask inEventMask, void* inContext);

	static void RemoveCallback( SecKeychainCallback inCallbackFunction );
    //static void RemoveCallbackUPP(KCCallbackUPP inCallbackFunction);
	static bool HasCallbacks()
	{ return CCallbackMgr::Instance().mEventCallbacks.size() > 0; };
	
private:

	void consume (SecurityServer::NotificationDomain domain, SecurityServer::NotificationEvent whichEvent,
				  const CssmData &data);
	
    void AlertClients(const list<CallbackInfo> &eventCallbacks, SecKeychainEvent inEvent, pid_t inPid,
							 const Keychain& inKeychain, const Item &inItem);

    // Use these as a CFRunLoop callback
    static void tellClient(CFRunLoopTimerRef timer, void* ctx);
    static void cfrunLoopActive(CFRunLoopTimerRef timer, void* info);

    bool initialized() { return mInitialized; }

	list<CallbackInfo> 		mEventCallbacks;
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_CCALLBACKMGR_H_
