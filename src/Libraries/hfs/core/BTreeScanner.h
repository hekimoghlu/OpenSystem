/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE
#include <sys/time.h>

#include "FileMgrInternal.h"
#include "BTreesPrivate.h"

// amount of time we are allowed to process a catalog search (in µ secs)
// NOTE - code assumes kMaxMicroSecsInKernel is less than 1,000,000
enum { kMaxMicroSecsInKernel = (1000 * 100) };	// 1 tenth of a second

// btree node scanner buffer size.  at 32K we get 8 nodes.  this is the size used
// in Mac OS 9
enum { kCatSearchBufferSize = (32 * 1024) };


/*
 * ============ W A R N I N G ! ============
 * DO NOT INCREASE THE SIZE OF THIS STRUCT!
 * It must be less than or equal to the size of 
 * the opaque searchstate struct (in sys/attr.h).
 */
/* Private description used in hfs_search */
struct CatPosition 
{
  u_int32_t		writeCount;    	/* The BTree's write count (to see if the catalog writeCount */
                              	/* changed since the last search).  If 0, the rest */
                             	/* of the record is invalid, start from beginning. */
  u_int32_t     nextNode;     	/* node number to resume search */
  u_int32_t   	nextRecord;  	/* record number to resume search */
  u_int32_t   	recordsFound; 	/* number of leaf records seen so far */
};
typedef struct CatPosition              CatPosition;


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
	struct buf *		bufferPtr;
	BTreeControlBlock *	btcb;
	
	//	The following fields are the dynamic state of the current scan.
	u_int32_t			nodeNum;			// zero is first node
	u_int32_t			recordNum;			// zero is first record
	BTNodeDescriptor *	currentNodePtr;		// points to current node within buffer
	u_int32_t			nodesLeftInBuffer;	// number of valid nodes still in the buffer
	u_int32_t			recordsFound;		// number of leaf records seen so far
	struct timeval		startTime;			// time we started catalog search
};
typedef struct BTScanState BTScanState;


/* *********************** PROTOTYPES *********************** */

int	BTScanInitialize(	const FCB *		btreeFile,
						u_int32_t		startingNode,
						u_int32_t		startingRecord,
						u_int32_t		recordsFound,
						u_int32_t		bufferSize,
						BTScanState	*	scanState     );
							
int BTScanNextRecord(	BTScanState *	scanState,
						Boolean			avoidIO,
						void * *		key,
						void * *		data,
						u_int32_t *		dataSize  );

int	BTScanTerminate(	BTScanState *	scanState,
						u_int32_t *		startingNode,
						u_int32_t *		startingRecord,
						u_int32_t *		recordsFound	);

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* !_BTREESCANNER_H_ */
