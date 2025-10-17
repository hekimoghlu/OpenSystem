/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#ifndef	_BTREESCANNER_H_
#define _BTREESCANNER_H_

#include "BTreePrivate.h"

// btree node scanner buffer size.  Joe Sokol suggests 128K as a max (2002 WWDC)
enum { kCatScanBufferSize = (128 * 1024) };


/*
	BTScanState - This structure is used to keep track of the current state
	of a BTree scan.  It contains both the dynamic state information (like
	the current node number and record number) and information that is static
	for the duration of a scan (such as buffer pointers).
	
	NOTE: recordNum may equal or exceed the number of records in the node
	number nodeNum.  If so, then the next attempt to get a record will move
	to a new node number.
*/
struct BTScanState 
{
	//	The following fields are set up once at initialization time.
	//	They are not changed during a scan.
	u_int32_t			bufferSize;
	void *				bufferPtr;
	BTreeControlBlock *	btcb;
	
	//	The following fields are the dynamic state of the current scan.
	u_int32_t			nodeNum;			// zero is first node
	u_int32_t			recordNum;			// zero is first record
	BTNodeDescriptor *	currentNodePtr;		// points to current node within buffer
	int32_t				nodesLeftInBuffer;	// number of valid nodes still in the buffer
	int64_t				recordsFound;		// number of leaf records seen so far
};
typedef struct BTScanState BTScanState;


/* *********************** PROTOTYPES *********************** */

int	BTScanInitialize(	const SFCB *	btreeFile,
							BTScanState	*	scanState     );
							
int BTScanNextRecord(	BTScanState *	scanState,
						void * *		key,
						void * *		data,
						u_int32_t *		dataSize  );

int	BTScanTerminate(	BTScanState *	scanState	);

#endif /* !_BTREESCANNER_H_ */
