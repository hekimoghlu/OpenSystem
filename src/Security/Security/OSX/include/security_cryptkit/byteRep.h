/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef	_CK_BYTEREP_H_
#define _CK_BYTEREP_H_

#include "feeTypes.h"
#include "giantIntegers.h"
#include "elliptic.h"
#include "curveParams.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Support for bytestream key and signature representation.
 */
int intToByteRep(int i, unsigned char *buf);
int shortToByteRep(short s, unsigned char *buf);
int giantToByteRep(giant g, unsigned char *buf);
int keyToByteRep(key k, unsigned char *buf);
int curveParamsToByteRep(curveParams *cp, unsigned char *buf);
int sigToByteRep(int magic,
	int version,
	int minVersion,
	giant g0,
	giant g1,
	unsigned char *buf);

int lengthOfByteRepGiant(giant g);
int lengthOfByteRepKey(key k);
int lengthOfByteRepCurveParams(curveParams *cp);
int lengthOfByteRepSig(giant g0,
	giant g1);

int byteRepToInt(const unsigned char *buf);
unsigned short byteRepToShort(const unsigned char *buf);
giant byteRepToGiant(const unsigned char *buf,
	unsigned bufLen,
	unsigned *giantLen);
key byteRepToKey(const unsigned char *buf,
	unsigned bufLen,
	int twist,
	curveParams *cp,
	unsigned *keyLen);	// returned
curveParams *byteRepToCurveParams(const unsigned char *buf,
	unsigned bufLen,
	unsigned *cpLen);
int byteRepToSig(const unsigned char *buf,
	unsigned bufLen,
	int codeVersion,
	int *sigMagic,				// RETURNED
	int *sigVersion,			// RETURNED
	int *sigMinVersion,			// RETURNED
	giant *g0,					// alloc'd  & RETURNED
	giant *g1);					// alloc'd  & RETURNED

#ifdef __cplusplus
}
#endif

#endif	/*_CK_BYTEREP_H_*/
