/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
// cssmerrors
//
#include <security_cdsa_utilities/cssmerrors.h>
#include <security_utilities/mach++.h>
#include <Security/cssmapple.h>
#include <Security/SecBase.h>
#include <Security/SecBasePriv.h>

namespace Security {


CssmError::CssmError(CSSM_RETURN err, bool suppresslogging) : error(err)
{
    SECURITY_EXCEPTION_THROW_CSSM(this, err);

    if(!suppresslogging || secinfoenabled("security_exception")) {
        snprintf(whatBuffer, whatBufferSize, "CSSM Exception: %d %s", err, cssmErrorString(err));
        switch(err) {
            /* reduce log noise by filtering out some non-error exceptions */
            case CSSMERR_CL_UNKNOWN_TAG:
                break;
            default:
                secnotice("security_exception", "%s", what());
                LogBacktrace();
                break;
        }
    }
}


const char *CssmError::what() const _NOEXCEPT
{
    return whatBuffer;
}


OSStatus CssmError::osStatus() const
{
	if (error == CSSM_ERRCODE_INVALID_POINTER)
	{
		return errSecParam;
	}

	return error;
}


int CssmError::unixError() const
{
	OSStatus err = osStatus();

	// embedded UNIX errno values are returned verbatim
	if (err >= errSecErrnoBase && err <= errSecErrnoLimit)
		return err - errSecErrnoBase;

	// re-map certain CSSM errors
    switch (err) {
	case CSSM_ERRCODE_MEMORY_ERROR:
		return ENOMEM;
	case CSSMERR_APPLEDL_DISK_FULL:
		return ENOSPC;
	case CSSMERR_APPLEDL_QUOTA_EXCEEDED:
		return EDQUOT;
	case CSSMERR_APPLEDL_FILE_TOO_BIG:
		return EFBIG;
	default:
		// cannot map this to errno space
		return -1;
    }
}


void CssmError::throwMe(CSSM_RETURN err)
{
	throw CssmError(err, false);
}

void CssmError::throwMeNoLogging(CSSM_RETURN err)
{
    throw CssmError(err, true);
}


CSSM_RETURN CssmError::merge(CSSM_RETURN error, CSSM_RETURN base)
{
	if (0 < error && error < CSSM_ERRORCODE_COMMON_EXTENT) {
		return base + error;
	} else {
		return error;
	}
}

//
// Get a CSSM_RETURN from a CommonError
//
CSSM_RETURN CssmError::cssmError(const CommonError &error, CSSM_RETURN base)
{
	if (const CssmError *cssm = dynamic_cast<const CssmError *>(&error)) {
		return cssmError(cssm->error, base);
	} else if (const MachPlusPlus::Error *mach = dynamic_cast<const MachPlusPlus::Error *>(&error)) {
		switch (mach->error) {
		case BOOTSTRAP_UNKNOWN_SERVICE:
		case MIG_SERVER_DIED:
			return CSSM_ERRCODE_SERVICE_NOT_AVAILABLE;
        case MIG_BAD_ID:
            return CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED;
		default:
			return CSSM_ERRCODE_INTERNAL_ERROR;
		}
	} else {
		return error.osStatus();
	}
}

CSSM_RETURN CssmError::cssmError(CSSM_RETURN error, CSSM_RETURN base)
{
    if (0 < error && error < CSSM_ERRORCODE_COMMON_EXTENT) {
        return base + error;
    } else {
        return error;
    }
}


}   // namespace Security
