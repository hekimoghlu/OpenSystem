/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#include "BTreesPrivate.h"
#include <sys/kernel.h>
#include <libkern/libkern.h>


// local routines
static OSErr	CheckBTreeKey(const BTreeKey *key, const BTreeControlBlock *btcb);

#if DEBUG
static Boolean	ValidHFSRecord(const void *record, const BTreeControlBlock *btcb, u_int16_t recordSize);
#endif

OSErr ReplaceBTreeRecord(FileReference refNum, const void* key, u_int32_t hint, void *newData, u_int16_t dataSize, u_int32_t *newHint)
{
	FSBufferDescriptor	btRecord;
	struct BTreeIterator *iterator = NULL;
	FCB					*fcb;
	BTreeControlBlock	*btcb;
	OSStatus			result;

	iterator = hfs_malloc_type(struct BTreeIterator);

	fcb = GetFileControlBlock(refNum);
	btcb = (BTreeControlBlock*) fcb->fcbBTCBPtr;

	btRecord.bufferAddress = newData;
	btRecord.itemSize = dataSize;
	btRecord.itemCount = 1;

	iterator->hint.nodeNum = hint;

	result = CheckBTreeKey((const BTreeKey *) key, btcb);
	if (result) {
		goto ErrorExit;
	}

	BlockMoveData(key, &iterator->key, CalcKeySize(btcb, (const BTreeKey *) key));		//€€ should we range check against maxkeylen?

#if DEBUG
	if ( !ValidHFSRecord(newData, btcb, dataSize) )
		DebugStr("ReplaceBTreeRecord: bad record?");
#endif

	result = BTReplaceRecord( fcb, iterator, &btRecord, dataSize );

	*newHint = iterator->hint.nodeNum;

ErrorExit:

	hfs_free_type(iterator, struct BTreeIterator);
	return result;
}



static OSErr CheckBTreeKey(const BTreeKey *key, const BTreeControlBlock *btcb)
{
	u_int16_t	keyLen;
	
	if ( btcb->attributes & kBTBigKeysMask )
		keyLen = key->length16;
	else
		keyLen = key->length8;

	if ( (keyLen < 6) || (keyLen > btcb->maxKeyLength) )
	{
		hfs_debug("CheckBTreeKey: bad key length!");
		return fsBTInvalidKeyLengthErr;
	}
	
	return noErr;
}

#if DEBUG

static Boolean ValidHFSRecord(const void *record, const BTreeControlBlock *btcb, u_int16_t recordSize)
{
	u_int32_t			cNodeID;
	
	if (btcb->maxKeyLength == kHFSPlusExtentKeyMaximumLength )
	{
		return ( recordSize == sizeof(HFSPlusExtentRecord) );
	} else { // Catalog record
		const CatalogRecord *catalogRecord = (const CatalogRecord*) record;

		switch(catalogRecord->recordType)
		{
			case kHFSPlusFolderRecord:
			{
				if ( recordSize != sizeof(HFSPlusCatalogFolder) )
					return false;
				if ( catalogRecord->hfsPlusFolder.flags != 0 )
					return false;
				if ( catalogRecord->hfsPlusFolder.valence > 0x7FFF )
					return false;
					
				cNodeID = catalogRecord->hfsPlusFolder.folderID;
	
				if ( (cNodeID == 0) || (cNodeID < 16 && cNodeID > 2) )
					return false;
			}
			break;
		
			case kHFSPlusFileRecord:
			{
//				u_int16_t					i;
				const HFSPlusExtentDescriptor	*dataExtent;
				const HFSPlusExtentDescriptor	*rsrcExtent;
				
				if ( recordSize != sizeof(HFSPlusCatalogFile) )
					return false;								
				if ( (catalogRecord->hfsPlusFile.flags & ~(0x83)) != 0 )
					return false;
					
				cNodeID = catalogRecord->hfsPlusFile.fileID;
				
				if ( cNodeID < 16 )
					return false;
		
				// make sure 0 ¾ LEOF ¾ PEOF for both forks
		
				dataExtent = (const HFSPlusExtentDescriptor*) &catalogRecord->hfsPlusFile.dataFork.extents;
				rsrcExtent = (const HFSPlusExtentDescriptor*) &catalogRecord->hfsPlusFile.resourceFork.extents;
	
#if 0
				for (i = 0; i < kHFSPlusExtentDensity; ++i)
				{
					if ( (dataExtent[i].blockCount > 0) && (dataExtent[i].startBlock == 0) )
						return false;
					if ( (rsrcExtent[i].blockCount > 0) && (rsrcExtent[i].startBlock == 0) )
						return false;
				}
#endif
			}
			break;		

			case kHFSPlusFileThreadRecord:
			case kHFSPlusFolderThreadRecord:
			{
				if ( recordSize > sizeof(HFSPlusCatalogThread) || recordSize < (sizeof(HFSPlusCatalogThread) - sizeof(HFSUniStr255)))
					return false;
	
				cNodeID = catalogRecord->hfsPlusThread.parentID;
				if ( (cNodeID == 0) || (cNodeID < 16 && cNodeID > 2) )
					return false;
							
				if ( (catalogRecord->hfsPlusThread.nodeName.length == 0) ||
					 (catalogRecord->hfsPlusThread.nodeName.length > 255) )
					return false;
			}
			break;

			default:
				return false;
		}
	}
	
	return true;	// record appears to be OK
}

#endif // DEBUG
