/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#ifndef __EAPTLS_UI_H__
#define __EAPTLS_UI_H__


/* type of request from backend to UI */
#define REQUEST_TRUST_EVAL	1

/* type of response from UI to back end */
#define RESPONSE_OK			0
#define RESPONSE_ERROR		1
#define RESPONSE_CANCEL		2


typedef struct eaptls_ui_ctx 
{
    u_int16_t	len;			/* length of this context */
    u_int16_t	id;				/* generation id, to match request/response */
    u_int16_t	request;        /* type of request from backend to UI */
    u_int16_t	response;       /* type of request from backend to UI */
} eaptls_ui_ctx;


int eaptls_ui_load(CFBundleRef bundle, void *logdebug, void *logerror);
void eaptls_ui_dispose(void);

int eaptls_ui_trusteval(CFDictionaryRef publishedProperties, 
					void *data_in, int data_in_len,
                    void **data_out, int *data_out_len);


#endif
