/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#include <string>
#include <exception>

#include <dlfcn.h>
#include <dispatch/dispatch.h>
#include <CoreServices/CoreServicesPriv.h>

#include <security_utilities/cfutilities.h>
#include <security_utilities/unix++.h>
#include <security_utilities/logging.h>

#include "SecTranslocate.h"
#include "SecTranslocateLSNotification.hpp"
#include "SecTranslocateUtilities.hpp"
#include "SecTranslocateShared.hpp"

#define LS_FRAMEWORK_PATH "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/LaunchServices"

namespace Security {
namespace SecTranslocate {

/* Types for the LaunchServices symbols that I am Pseudo weak linking */
typedef void (^LSNotificationHandler_t) (LSNotificationCode, CFAbsoluteTime, CFTypeRef, LSASNRef, LSSessionID, LSNotificationID);
typedef LSNotificationID (*ScheduleNotificationOnQueueWithBlock_t) (LSSessionID, CFTypeRef, dispatch_queue_t,LSNotificationHandler_t);
typedef OSStatus (*ModifyNotification_t)(LSNotificationID, UInt32, const LSNotificationCode *, UInt32, const LSNotificationCode *, CFTypeRef, CFTypeRef);
typedef OSStatus (*UnscheduleNotificationFunction_t)(LSNotificationID);
typedef LSASNRef (*ASNCreateWithPid_t)(CFAllocatorRef, int);
typedef uint64_t (*ASNToUInt64_t)(LSASNRef);
typedef Boolean (*IsApplicationRunning_t)(LSSessionID, LSASNRef);

/* Class to contain all the Launch Services functions I need to weak link */
class LaunchServicesProxy
{
public:
    static LaunchServicesProxy* get();

    inline LSNotificationID scheduleNotificationOnQueueWithBlock (LSSessionID s, CFTypeRef r, dispatch_queue_t q,LSNotificationHandler_t b) const
        { return pScheduleNotificationOnQueueWithBlock ? pScheduleNotificationOnQueueWithBlock(s,r,q,b) : (LSNotificationID)kLSNotificationInvalidID; };
    inline OSStatus modifyNotification(LSNotificationID i, UInt32 c, const LSNotificationCode * s, UInt32 l, const LSNotificationCode *n, CFTypeRef a, CFTypeRef r) const
        { return pModifyNotification ? pModifyNotification(i,c,s,l,n,a,r) : kLSUnknownErr; };
    inline OSStatus unscheduleNotificationFunction(LSNotificationID i) const
        { return pUnscheduleNotificationFunction ? pUnscheduleNotificationFunction(i) : kLSUnknownErr; };
    inline LSASNRef asnCreateWithPid(CFAllocatorRef a, int p) const
        { return pASNCreateWithPid ? pASNCreateWithPid(a,p) : NULL; };
    inline uint64_t asnToUInt64 (LSASNRef a) const
        { return pASNToUInt64 ? pASNToUInt64(a) : 0; };
    inline CFStringRef bundlePathKey() const { return pBundlePathKey ? *pBundlePathKey : NULL;};
    inline Boolean isApplicationRunning(LSSessionID i, LSASNRef a) const {return pIsApplicationRunning ? pIsApplicationRunning(i,a): false;};

private:
    LaunchServicesProxy();

    void* handle;
    ScheduleNotificationOnQueueWithBlock_t pScheduleNotificationOnQueueWithBlock;
    ModifyNotification_t pModifyNotification;
    UnscheduleNotificationFunction_t pUnscheduleNotificationFunction;
    ASNCreateWithPid_t pASNCreateWithPid;
    ASNToUInt64_t pASNToUInt64;
    CFStringRef *pBundlePathKey;
    IsApplicationRunning_t pIsApplicationRunning;
};

/* resolve all the symbols. Throws if something isn't found. */
LaunchServicesProxy::LaunchServicesProxy()
{
    handle = checkedDlopen(LS_FRAMEWORK_PATH, RTLD_LAZY | RTLD_NOLOAD);

    pScheduleNotificationOnQueueWithBlock = (ScheduleNotificationOnQueueWithBlock_t) checkedDlsym(handle, "_LSScheduleNotificationOnQueueWithBlock");
    pModifyNotification = (ModifyNotification_t) checkedDlsym(handle, "_LSModifyNotification");
    pUnscheduleNotificationFunction = (UnscheduleNotificationFunction_t) checkedDlsym(handle, "_LSUnscheduleNotificationFunction");
    pASNCreateWithPid = (ASNCreateWithPid_t) checkedDlsym(handle, "_LSASNCreateWithPid");
    pASNToUInt64 = (ASNToUInt64_t) checkedDlsym(handle, "_LSASNToUInt64");
    pBundlePathKey = (CFStringRef*) checkedDlsym(handle, "_kLSBundlePathKey");
    pIsApplicationRunning = (IsApplicationRunning_t) checkedDlsym(handle, "_LSIsApplicationRunning");
}

/* Singleton getter for the proxy */
LaunchServicesProxy* LaunchServicesProxy::get()
{
    static dispatch_once_t initialized;
    static LaunchServicesProxy* me = NULL;
    __block exception_ptr exception(0);

    dispatch_once(&initialized, ^{
        try
        {
            me = new LaunchServicesProxy();
        }
        catch (...)
        {
            Syslog::critical("SecTranslocate: error while creating LaunchServicesProxy");
            exception = current_exception();
        }
    });

    if (me == NULL)
    {
        if(exception)
        {
            rethrow_exception(exception); //already logged in this case
        }
        else
        {
            Syslog::critical("SecTranslocate: LaunchServicesProxy initialization has failed");
            UnixError::throwMe(EINVAL);
        }
    }

    return me;
}

/* Save the notification queue so we can do things async later */
LSNotificationMonitor::LSNotificationMonitor(dispatch_queue_t q): notificationQ(q)
{
    if (notificationQ == NULL)
    {
        Syslog::critical("SecTranslocate::LSNotificationMonitor initialized without a queue.");
        UnixError::throwMe(EINVAL);
    }
    
    dispatch_retain(notificationQ);
}

/* Release the dispatch queue if this ever gets destroyed */
LSNotificationMonitor::~LSNotificationMonitor()
{
    dispatch_release(notificationQ);
}

/*  Check to see if a path is translocated. If it isn't or no path is provided then return 
    an empty string. If it is, return the path as a c++ string. */
string LSNotificationMonitor::stringIfTranslocated(CFStringRef appPath)
{
    if(appPath == NULL)
    {
        Syslog::error("SecTranslocate: no appPath provided");
        return "";
    }

    CFRef<CFURLRef> appURL = makeCFURL(appPath);
    bool isTranslocated = false;

    string out = cfString(appURL);

    if (!SecTranslocateIsTranslocatedURL(appURL, &isTranslocated, NULL))
    {
        Syslog::error("SecTranslocate: path for asn doesn't exist or isn't accessible: %s",out.c_str());
        return "";
    }

    if(!isTranslocated)
    {
        Syslog::error("SecTranslocate: asn is not translocated: %s",out.c_str());
        return "";
    }

    return out;
}

/* register for a notification about the death of the requested PID with launch services if the pid is translocated */
void LSNotificationMonitor::checkIn(pid_t pid)
{
    dispatch_async(notificationQ, ^(){
        try
        {
            LaunchServicesProxy* lsp = LaunchServicesProxy::get();

            CFRef<LSASNRef> asn = lsp->asnCreateWithPid(kCFAllocatorDefault, pid);

            if(lsp->isApplicationRunning(kLSDefaultSessionID, asn))
            {
                LSNotificationID nid = lsp->scheduleNotificationOnQueueWithBlock(kLSDefaultSessionID,
                                                                                 cfEmptyArray(),
                                                                                 notificationQ,
                                                                                 ^ (LSNotificationCode notification,
                                                                                    CFAbsoluteTime notificationTime,
                                                                                    CFTypeRef dataRef,
                                                                                    LSASNRef affectedASNRef,
                                                                                    LSSessionID session,
                                                                                    LSNotificationID notificationID){
                    if( notification == kLSNotifyApplicationDeath && dataRef)
                    {
                        this->asnDied(dataRef);
                    }

                    lsp->unscheduleNotificationFunction(notificationID);
                });
                LSNotificationCode notificationCode = kLSNotifyApplicationDeath;
                lsp->modifyNotification(nid, 1, &notificationCode, 0, NULL, asn, NULL);
            }
            else
            {
                Syslog::warning("SecTranslocate: pid %d checked in, but it is not running",pid);
            }
        }
        catch(...)
        {
            Syslog::error("SecTranslocate: checkin failed for pid %d",pid);
        }
    });
}

/* use the supplied dictionary to perform volume cleanup. If the dictionary contains a bundle path
  and that bundle path still exists and is translocated, then unmount it. Otherwise trigger a
  unmount of any translocation point that doesn't point to an existant volume. */
void LSNotificationMonitor::asnDied(CFTypeRef data) const
{
    string path;
    try
    {
        CFDictionaryRef dict = NULL;
        if(CFGetTypeID(data) == CFDictionaryGetTypeID())
        {
            dict = (CFDictionaryRef)data;
        }
        else
        {
            Syslog::error("SecTranslocate: no data dictionary at app death");
            return;
        }

        LaunchServicesProxy* lsp = LaunchServicesProxy::get();
        path = stringIfTranslocated((CFStringRef)CFDictionaryGetValue(dict,lsp->bundlePathKey()));
    }
    catch(...)
    {
        Syslog::error("SecTranslocate: asn death processing failed");
        return;
    }

    /* wait 5 seconds after death */
    dispatch_time_t when = dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC);

    dispatch_after(when, notificationQ, ^() {
        try
        {
            if(path.empty())
            {
                /* we got an asn death notification but the path either wasn't translocated or didn't exist
                   in case it didn't exist try to clean up stale translocation points.
                   Calling this function with no parameter defaults to a blank volume which causes
                   only translocation points that point to non-existant volumes to be cleaned up. */
                destroyTranslocatedPathsForUserOnVolume();
            }
            else
            {
                /* remove the translocation point for the app */
                destroyTranslocatedPathForUser(path);
            }
        }
        catch(...)
        {
            Syslog::error("SecTranslocate: problem deleting translocation after app death: %s", path.c_str());
        }
    });
}

} //namespace SecTranslocate
} //namespace Security
