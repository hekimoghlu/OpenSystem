/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
// AIFFSupport.c created by CJS on Wed 10-May-2000

#ifndef __AIFF_SUPPORT_H__
#include "AIFFSupport.h"
#endif

#ifndef __APPLE_CDDA_FS_DEBUG_H__
#include "AppleCDDAFileSystemDebug.h"
#endif

#ifndef __APPLE_CDDA_FS_VNODE_OPS_H__
#include "AppleCDDAFileSystemVNodeOps.h"
#endif


// System Includes
#include <sys/systm.h>
#include <libkern/OSByteOrder.h>


#if defined(__LITTLE_ENDIAN__)

// Conversion Routines
static void
SwapContainerChunk ( ContainerChunk * chunk );

static void
SwapFormatVersionChunk ( FormatVersionChunk * chunk );

static void
SwapExtCommonChunk ( ExtCommonChunk * chunk );

static void
SwapSoundDataChunk ( SoundDataChunk * chunk );

static void
SwapCDAIFFHeader ( CDAIFFHeader * header );

static void
SwapFloat80	( Float80 * value );


//-----------------------------------------------------------------------------
// 	SwapContainerChunk	-	This converts a little endian representation
//									of a ContainerChunk into a big endian
//									representation
//-----------------------------------------------------------------------------

static void
SwapContainerChunk ( ContainerChunk * chunk )
{
	
	chunk->ckID 	= OSSwapInt32 ( chunk->ckID );
	chunk->ckSize	= OSSwapInt32 ( chunk->ckSize );
	chunk->formType	= OSSwapInt32 ( chunk->formType );
	
}


//-----------------------------------------------------------------------------
// 	SwapContainerChunk	-	This converts a little endian representation
//										of a FormatVersionChunk into a big endian
//										representation
//-----------------------------------------------------------------------------

static void
SwapFormatVersionChunk ( FormatVersionChunk * chunk )
{
	
	chunk->ckID 		= OSSwapInt32 ( chunk->ckID );
	chunk->ckSize		= OSSwapInt32 ( chunk->ckSize );
	chunk->timeStamp	= OSSwapInt32 ( chunk->timeStamp );
	
}


//-----------------------------------------------------------------------------
// 	SwapExtCommonChunk	-	This converts a little endian representation
//									of an ExtCommonChunk into a big endian
//									representation
//-----------------------------------------------------------------------------

static void
SwapExtCommonChunk ( ExtCommonChunk * chunk )
{
	
	SwapFloat80 ( &chunk->sampleRate );
	
	chunk->ckID				= OSSwapInt32 ( chunk->ckID );
	chunk->ckSize			= OSSwapInt32 ( chunk->ckSize );
	chunk->numChannels		= OSSwapInt16 ( chunk->numChannels );
	chunk->numSampleFrames	= OSSwapInt32 ( chunk->numSampleFrames );
	chunk->sampleSize		= OSSwapInt16 ( chunk->sampleSize );
	chunk->compressionType	= OSSwapInt32 ( chunk->compressionType );
	
}


//-----------------------------------------------------------------------------
// 	SwapSoundDataChunk	-	This converts a little endian representation
//									of a SoundDataChunk into a big endian
//									representation
//-----------------------------------------------------------------------------

static void
SwapSoundDataChunk ( SoundDataChunk * chunk )
{
	
	chunk->ckID 		= OSSwapInt32 ( chunk->ckID );
	chunk->ckSize 		= OSSwapInt32 ( chunk->ckSize );
	chunk->offset		= OSSwapInt32 ( chunk->offset );
	chunk->blockSize	= OSSwapInt32 ( chunk->blockSize );
		
}


//-----------------------------------------------------------------------------
// 	SwapCDAIFFHeader	-	This converts a little endian representation
//								of a CDAIFFHeader into a big endian
//								representation
//-----------------------------------------------------------------------------

static void
SwapCDAIFFHeader ( CDAIFFHeader * header )
{
	
	SwapContainerChunk ( &header->containerChunk );
	SwapFormatVersionChunk ( &header->formatVersionChunk );
	SwapExtCommonChunk ( &header->commonChunk );
	SwapSoundDataChunk ( &header->soundDataChunk );
	
}


//-----------------------------------------------------------------------------
// 	SwapFloat80	-	This converts a little endian representation
//							of a Float80 into a big endian representation
//-----------------------------------------------------------------------------

static void
SwapFloat80	( Float80 * data )
{
	
	data->exp 		= OSSwapInt16 ( data->exp );
	data->man[0]	= OSSwapInt16 ( data->man[0] );
	data->man[1]	= OSSwapInt16 ( data->man[1] );
	data->man[2]	= OSSwapInt16 ( data->man[2] );
	data->man[3]	= OSSwapInt16 ( data->man[3] );
	
}


#endif /* defined(__LITTLE_ENDIAN__) */


//-----------------------------------------------------------------------------
// 	BuildCDAIFFHeader	-	This routine builds a CDAIFFHeader and explicitly
//							makes it Big Endian (as defined by AIFF standard)
//-----------------------------------------------------------------------------

void
BuildCDAIFFHeader ( CDAIFFHeader * header, uint32_t fileSize )
{
	
	ExtCommonChunk *		commonChunkPtr 			= NULL;
	SoundDataChunk *		soundDataChunkPtr		= NULL;
	ContainerChunk *		containerChunkPtr		= NULL;
	FormatVersionChunk *	formatVersionChunkPtr	= NULL;
	uint32_t				dataSize				= fileSize - kPhysicalMediaBlockSize;
	
	// Get the address of each sub-structure to make this easier to read
	commonChunkPtr			= &header->commonChunk;
	soundDataChunkPtr		= &header->soundDataChunk;
	containerChunkPtr		= &header->containerChunk;
	formatVersionChunkPtr	= &header->formatVersionChunk;
	
	// Setup the version chunk
	formatVersionChunkPtr->ckID			= kFormatVersionID;
	formatVersionChunkPtr->ckSize		= ( uint32_t )( sizeof ( FormatVersionChunk ) - sizeof ( ChunkHeader ) );
	formatVersionChunkPtr->timeStamp	= kAIFCVersion1;
	
	// Setup the common chunk
	commonChunkPtr->ckID 				= kCommonID;
	commonChunkPtr->ckSize 				= ( uint32_t )( sizeof ( ExtCommonChunk ) - sizeof ( ChunkHeader ) );
	commonChunkPtr->numChannels 		= kStereo;			// 2 channels
	commonChunkPtr->numSampleFrames 	= dataSize / 4; 	// 4 = ( k16BitsPerSample / 8 ) * kStereo
	commonChunkPtr->sampleSize 			= k16BitsPerSample;	// Set the sample size to 16 bits
	commonChunkPtr->sampleRate.exp 		= 0x400E;			// Set the sample rate to 44.1 KHz
	commonChunkPtr->sampleRate.man[0] 	= 0xAC44;			// Set the sample rate to 44.1 KHz
	commonChunkPtr->sampleRate.man[1]	= 0x0000;			// Set the sample rate to 44.1 KHz	
	commonChunkPtr->sampleRate.man[2]	= 0x0000;			// Set the sample rate to 44.1 KHz
	commonChunkPtr->sampleRate.man[3]	= 0x0000;			// Set the sample rate to 44.1 KHz
	
	// Data streamed off disc is in little endian format
	commonChunkPtr->compressionType	= k16BitLittleEndianFormat;
	
	// Setup the soundData chunk
	soundDataChunkPtr->ckID 		= kSoundDataID;
	soundDataChunkPtr->offset 		= ( uint32_t ) ( kPhysicalMediaBlockSize - sizeof ( CDAIFFHeader ) );
	soundDataChunkPtr->ckSize 		= ( uint32_t ) ( ( sizeof ( SoundDataChunk ) - sizeof ( ChunkHeader ) ) +
									  dataSize + soundDataChunkPtr->offset );
	soundDataChunkPtr->blockSize 	= 0;
	
	// Setup the container chunk
	containerChunkPtr->ckID 	= kFormID;
	containerChunkPtr->ckSize 	= ( uint32_t )
								  ( ( sizeof ( ContainerChunk ) - sizeof ( ChunkHeader ) ) +		// size of container chunk variables
					   			    ( formatVersionChunkPtr->ckSize + sizeof ( ChunkHeader ) ) + 	// size of common chunk
					   			    ( commonChunkPtr->ckSize + sizeof ( ChunkHeader ) ) +			// size of common chunk
					   			    ( soundDataChunkPtr->ckSize + sizeof ( ChunkHeader ) ) );		// size of sound data chunk
	
	// Save as uncompressed AIFF-C
	containerChunkPtr->formType = kAIFCID;
	
#if defined(__LITTLE_ENDIAN__)
	
	// Convert from natural byte order to big endian byte order
	// because AIFF Header data MUST be big endian (the audio data
	// doesn't necessarily have to)
	SwapCDAIFFHeader ( header );
	
#endif /* defined(__LITTLE_ENDIAN__) */
	
}


//-----------------------------------------------------------------------------
//				End				Of			File
//-----------------------------------------------------------------------------
