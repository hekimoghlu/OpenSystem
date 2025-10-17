/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#ifndef _XDR_DLDB_H
#define _XDR_DLDB_H

#include "sec_xdr.h"
#include "xdr_cssm.h"

#include <security_cdsa_utilities/cssmdb.h>

bool_t xdr_DLDbFlatIdentifier(XDR *xdrs, DataWalkers::DLDbFlatIdentifier *objp);
bool_t xdr_DLDbFlatIdentifierRef(XDR *xdrs, DataWalkers::DLDbFlatIdentifier **objp);

class CopyIn {
    public:
        CopyIn(const void *data, xdrproc_t proc) : mLength(0), mData(0) {
            if (data && !::copyin(static_cast<uint8_t*>(const_cast<void *>(data)), proc, &mData, &mLength))
                CssmError::throwMe(CSSM_ERRCODE_MEMORY_ERROR);  
        }
        ~CopyIn() { if (mData) free(mData);     }
        u_int length() { return mLength; }
        void *data() { return mData; }
    protected:
        u_int mLength;
        void *mData;
};

// split out CSSM_DATA variant
class CopyOut {
    public:
		// CSSM_DATA can be output only if empty, but also specify preallocated memory to use
        CopyOut(void *copy, size_t size, xdrproc_t proc, bool dealloc = false, CSSM_DATA *in_out_data = NULL) : mLength(in_out_data?(u_int)in_out_data->Length:0), mData(NULL), mInOutData(in_out_data), mDealloc(dealloc), mSource(copy), mSourceLen(size) {
            if (copy && size && !::copyout(copy, (u_int)size, proc, mInOutData ? reinterpret_cast<void**>(&mInOutData) : &mData, &mLength)) {
                if (mInOutData && mInOutData->Length) // DataOut behaviour: error back to user if likely related to amount of space passed in
                    CssmError::throwMe(CSSMERR_CSP_OUTPUT_LENGTH_ERROR);
                else
                    CssmError::throwMe(CSSM_ERRCODE_MEMORY_ERROR);
            }
        }
        ~CopyOut();
        u_int length() { return mLength; } 
        void* data() { return mData; }
        void* get() { void *tmp = mData; mData = NULL; return tmp; }
    protected:
        u_int mLength;
        void *mData;
		CSSM_DATA *mInOutData;
		bool mDealloc;
		void *mSource;
		size_t mSourceLen;
};

#endif /* !_XDR_AUTH_H */
