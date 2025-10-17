/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
 * sslRecord.c - Encryption, decryption and MACing of data
*/

#include <SecureTransport.h>
#include "ssl.h"
#include "sslRecord.h"
#include "sslMemory.h"
#include "sslContext.h"
#include "sslDebug.h"
#include "SSLRecordInternal.h"

#include <string.h>

#include <utilities/SecIOFormat.h>

/*
 * Lots of servers fail to provide closure alerts when they disconnect.
 * For now we'll just accept it as long as it occurs on a clean record boundary
 * (and the handshake is complete).
 */
#define SSL_ALLOW_UNNOTICED_DISCONNECT	1


static OSStatus errorTranslate(int recordErr)
{
    switch(recordErr) {
        case errSecSuccess:
            return errSecSuccess;
        case errSSLRecordInternal:
            return errSSLInternal;
        case errSSLRecordWouldBlock:
            return errSSLWouldBlock;
        case errSSLRecordProtocol:
            return errSSLProtocol;
        case errSSLRecordNegotiation:
            return errSSLNegotiation;
        case errSSLRecordClosedAbort:
            return errSSLClosedAbort;
        case errSSLRecordConnectionRefused:
            return errSSLConnectionRefused;
        case errSSLRecordDecryptionFail:
            return errSSLDecryptionFail;
        case errSSLRecordBadRecordMac:
            return errSSLBadRecordMac;
        case errSSLRecordRecordOverflow:
            return errSSLRecordOverflow;
        case errSSLRecordUnexpectedRecord:
            return errSSLUnexpectedRecord;
        default:
            sslErrorLog("unknown error code returned in sslErrorTranslate: %d\n", recordErr);
            return recordErr;
    }
}

/* SSLWriteRecord
 *  Attempt to encrypt and queue an SSL record.
 */
OSStatus
SSLWriteRecord(SSLRecord rec, SSLContext *ctx)
{
    OSStatus    err;

    err=errorTranslate(ctx->recFuncs->write(ctx->recCtx, rec));

    switch(err) {
        case errSecSuccess:
            break;
        default:
            sslErrorLog("unexpected error code returned in SSLWriteRecord: %d\n", (int)err);
            break;
    }

    return err;
}

/* SSLFreeRecord
 *  Free a record returned by SSLReadRecord.
 */
OSStatus
SSLFreeRecord(SSLRecord rec, SSLContext *ctx)
{
    return ctx->recFuncs->free(ctx->recCtx, rec);
}

/* SSLReadRecord
 *  Attempt to read & decrypt an SSL record.
 *  Record content should be freed using SSLFreeRecord
 */
OSStatus
SSLReadRecord(SSLRecord *rec, SSLContext *ctx)
{
    return errorTranslate(ctx->recFuncs->read(ctx->recCtx, rec));
}

OSStatus SSLServiceWriteQueue(SSLContext *ctx)
{
    return errorTranslate(ctx->recFuncs->serviceWriteQueue(ctx->recCtx));
}
