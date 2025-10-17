/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
 * tls_record.h - Declarations of record layer callout struct to provide indirect calls to
 *     SSLv3 and TLS routines.
 */

#ifndef	_TLS_RECORD_INTERNAL_H_
#define _TLS_RECORD_INTERNAL_H_

#ifdef	__cplusplus
extern "C" {
#endif

// #include "sslRecord.h"

#include "sslTypes.h"
#include "sslMemory.h"
#include "SSLRecordInternal.h"
#include "sslContext.h"

#include <tls_record.h>

struct SSLRecordInternalContext;

typedef struct WaitingRecord
{   struct WaitingRecord    *next;
    size_t                  sent;
    /*
     * These two fields replace a dynamically allocated SSLBuffer;
     * the payload to write is contained in the variable-length
     * array data[].
     */
    size_t					length;
    uint8_t					data[1];
} WaitingRecord;


struct SSLRecordInternalContext
{
    tls_record_t        filter;

    /* Reference back to the SSLContext */
    SSLContextRef       sslCtx;

    /* buffering */
    SSLBuffer    		partialReadBuffer;
    size_t              amountRead;

    WaitingRecord       *recordWriteQueue;
};

#ifdef	__cplusplus
}
#endif

#endif 	/* _TLS_SSL_H_ */
