/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include "piddiskrep.h"
#include "sigblob.h"
#include <sys/param.h>
#include <sys/utsname.h>
#include <System/sys/codesign.h>
#include <libproc.h>
#include <xpc/xpc.h>

namespace Security {
namespace CodeSigning {
                
using namespace UnixPlusPlus;


void
PidDiskRep::setCredentials(const Security::CodeSigning::CodeDirectory *cd)
{
	// save the Info.plist slot
	if (cd->slotIsPresent(cdInfoSlot)) {
		mInfoPlistHash.take(makeCFData(cd->getSlot(cdInfoSlot, false), cd->hashSize));
	}
}

void
PidDiskRep::fetchData(void)
{
	if (mDataFetched)	// once
		return;
	
	xpc_connection_t conn = xpc_connection_create("com.apple.CodeSigningHelper",
						      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
	xpc_connection_set_event_handler(conn, ^(xpc_object_t object){ });
	xpc_connection_resume(conn);
	
	xpc_object_t request = xpc_dictionary_create(NULL, NULL, 0);
	assert(request != NULL);
	xpc_dictionary_set_string(request, "command", "fetchData");
	xpc_dictionary_set_int64(request, "pid", mPid);
	
	if (mAudit) {
		xpc_dictionary_set_data(request, "audit", mAudit.get(), sizeof(audit_token_t));
	}
	xpc_dictionary_set_data(request, "infohash", CFDataGetBytePtr(mInfoPlistHash), CFDataGetLength(mInfoPlistHash));
	
	xpc_object_t reply = xpc_connection_send_message_with_reply_sync(conn, request);
	if (reply && xpc_get_type(reply) == XPC_TYPE_DICTIONARY) {
		const void *data;
		size_t size;

		if (!mInfoPlist) {
			data = xpc_dictionary_get_data(reply, "infoPlist", &size);
			if (data && size > 0 && size < 50 * 1024)
				mInfoPlist.take(CFDataCreate(NULL, (const UInt8 *)data, (CFIndex)size));
		}
		if (!mBundleURL) {
			data = xpc_dictionary_get_data(reply, "bundleURL", &size);
			if (data && size > 0 && size < 50 * 1024)
				mBundleURL.take(CFURLCreateWithBytes(NULL, (const UInt8 *)data, (CFIndex)size, kCFStringEncodingUTF8, NULL));
		}
	}
	if (reply)
		xpc_release(reply);

	xpc_release(request);
	xpc_release(conn);
    
    if (!mBundleURL) {
        MacOSError::throwMe(errSecCSNoSuchCode);
    }

    mDataFetched = true;
}


PidDiskRep::PidDiskRep(pid_t pid, audit_token_t *audit, CFDataRef infoPlist)
	: mDataFetched(false)
{
        BlobCore header;
	
        mPid = pid;
        mInfoPlist = infoPlist;
    
        if (audit != NULL) {
            mAudit.reset(new audit_token_t);
            memcpy(mAudit.get(), audit, sizeof(audit_token_t));
        }
    
        //        fetchData();
    
        int rcent = EINVAL;
	
        if (audit != NULL) {
            rcent = ::csops_audittoken(pid, CS_OPS_BLOB, &header, sizeof(header), mAudit.get());
        } else {
            rcent = ::csops(pid, CS_OPS_BLOB, &header, sizeof(header));
        }
        if (rcent == 0)
            MacOSError::throwMe(errSecCSNoSuchCode);
        
        if (errno != ERANGE)
                UnixError::throwMe(errno);

        if (header.length() > 1024 * 1024)
                MacOSError::throwMe(errSecCSNoSuchCode);
        
        uint32_t bufferLen = (uint32_t)header.length();
        mBuffer = new uint8_t [bufferLen];
    
        if (audit != NULL) {
            UnixError::check(::csops_audittoken(pid, CS_OPS_BLOB, mBuffer, bufferLen, mAudit.get()));
        } else {
            UnixError::check(::csops(pid, CS_OPS_BLOB, mBuffer, bufferLen));
        }

        const EmbeddedSignatureBlob *b = (const EmbeddedSignatureBlob *)mBuffer;
        if (!b->validateBlob(bufferLen))
                MacOSError::throwMe(errSecCSSignatureInvalid);
}

PidDiskRep::~PidDiskRep()
{
        if (mBuffer)
                delete [] mBuffer;
}


bool PidDiskRep::supportInfoPlist()
{
		fetchData();
        return mInfoPlist;
}


CFDataRef PidDiskRep::component(CodeDirectory::SpecialSlot slot)
{
	if (slot == cdInfoSlot) {
		fetchData();
		return mInfoPlist.retain();
	}

	EmbeddedSignatureBlob *b = (EmbeddedSignatureBlob *)this->blob();
	return b->component(slot);
}

CFDataRef PidDiskRep::identification()
{
        return NULL;
}


CFURLRef PidDiskRep::copyCanonicalPath()
{
	fetchData();
	return mBundleURL.retain();
}

string PidDiskRep::recommendedIdentifier(const SigningContext &)
{
	return string("pid") + to_string(mPid);
}

size_t PidDiskRep::signingLimit()
{
        return 0;
}

size_t PidDiskRep::execSegLimit(const Architecture *)
{
		return 0;
}
    
string PidDiskRep::format()
{
        return "pid diskrep";
}

UnixPlusPlus::FileDesc &PidDiskRep::fd()
{
        UnixError::throwMe(EINVAL);
}

string PidDiskRep::mainExecutablePath()
{
        char path[MAXPATHLEN * 2];
        // This is unsafe by pid only, but so is using that path in general.
        if(::proc_pidpath(mPid, path, sizeof(path)) == 0)
		UnixError::throwMe(errno);

        return path;
}

bool PidDiskRep::appleInternalForcePlatform() const
{
	uint32_t flags = 0;
	int rcent = EINVAL;
	
	if (mAudit != NULL) {
		rcent = ::csops_audittoken(mPid, CS_OPS_STATUS, &flags, sizeof(flags),
								   mAudit.get());
	} else {
		rcent = ::csops(mPid, CS_OPS_STATUS, &flags, sizeof(flags));
	}
	
	if (rcent != 0) {
		MacOSError::throwMe(errSecCSNoSuchCode);
	}
	
	return (flags & CS_PLATFORM_BINARY) == CS_PLATFORM_BINARY;
}
                
} // end namespace CodeSigning
} // end namespace Security
