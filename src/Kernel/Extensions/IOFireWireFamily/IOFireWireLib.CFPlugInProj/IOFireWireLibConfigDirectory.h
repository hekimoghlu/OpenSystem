/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
 *  IOFireWireLibConfigDirectory.h
 *  IOFireWireFamily
 *
 *  Created by NWG on Thu Jan 18 2001.
 *  Copyright (c) 2000-2001 Apple Computer, Inc. All rights reserved.
 *
 */

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"

namespace IOFireWireLib {

	class Device ;
	class ConfigDirectory: public IOFireWireIUnknown
	{
		protected:
			typedef ::IOFireWireLibConfigDirectoryRef 	DirRef ;
	
		public:
			ConfigDirectory( const IUnknownVTbl & interface, Device& inUserClient, UserObjectHandle inDirRef ) ;
			ConfigDirectory( const IUnknownVTbl & interface, Device& inUserClient ) ;
			virtual ~ConfigDirectory() ;
			
			/*!
				@function update
				makes sure that the ROM has at least the specified capacity,
				and that the ROM is uptodate from its start to at least the
				specified quadlet offset.
				@result kIOReturnSuccess if the specified offset is now
				accessable at romBase[offset].
			*/
			IOReturn Update(UInt32 offset) ;
		
			/*!
				@function getKeyType
				Gets the data type for the specified key
				@param type on return, set to the data type
				@result kIOReturnSuccess if the key exists in the dictionary
			*/
			IOReturn GetKeyType(int key, IOConfigKeyType& type);
			
			/*!
				@function getKeyValue
				Gets the value for the specified key, in a variety of forms.
				@param value on return, set to the data type
				@param text if non-zero, on return points to the
				string description of the field, or NULL if no text found.
				@result kIOReturnSuccess if the key exists in the dictionary
				and is of a type appropriate for the value parameter
				@param value reference to variable to store the entry's value
			*/
			IOReturn GetKeyValue(int key, UInt32 &value, CFStringRef*& text);
			IOReturn GetKeyValue(int key, CFDataRef* value, CFStringRef*& text);
			IOReturn GetKeyValue(int key, DirRef& value, REFIID iid, CFStringRef*& text);
			IOReturn GetKeyOffset(int key, FWAddress& value, CFStringRef*& text);
			IOReturn GetKeyValue(int key, UserObjectHandle& value) ;
		
			/*!
				@function getIndexType
				Gets the data type for entry at the specified index
				@param type on return, set to the data type
				@result kIOReturnSuccess if the index exists in the dictionary
			*/
			IOReturn GetIndexType(int index, IOConfigKeyType &type);
			/*!
				@function getIndexKey
				Gets the key for entry at the specified index
				@param key on return, set to the key
				@result kIOReturnSuccess if the index exists in the dictionary
			*/
			IOReturn GetIndexKey(int index, int &key);
		
			/*!
				@function getIndexValue
				Gets the value at the specified index of the directory,
				in a variety of forms.
				@param type on return, set to the data type
				@result kIOReturnSuccess if the index exists in the dictionary
				and is of a type appropriate for the value parameter
				@param value reference to variable to store the entry's value
			*/
			IOReturn GetIndexValue(int index, UInt32& value);
			IOReturn GetIndexValue(int index, CFDataRef* value);
			IOReturn GetIndexValue(int index, CFStringRef* value);
//			IOReturn GetIndexValue(int index, UserObjectHandle& value) ;
			IOReturn GetIndexValue(int index, DirRef& value, REFIID iid);
		
			IOReturn GetIndexOffset(int index, FWAddress& value);
			IOReturn GetIndexOffset(int index, UInt32& value);
		
			/*!
				@function getIndexEntry
				Gets the entry at the specified index of the directory,
				as a raw UInt32.
				@param entry on return, set to the entry value
				@result kIOReturnSuccess if the index exists in the dictionary
				@param value reference to variable to store the entry's value
			*/
			IOReturn GetIndexEntry(int index, UInt32 &value);
		
			/*!
				@function getSubdirectories
				Creates an iterator over the subdirectories of the directory.
				@param iterator on return, set to point to an OSIterator
				@result kIOReturnSuccess if the iterator could be created
			*/
			IOReturn GetSubdirectories(io_iterator_t *outIterator);
		
			/*!
				@function getKeySubdirectories
				Creates an iterator over subdirectories of a given type of the directory.
				@param key type of subdirectory to iterate over
				@param iterator on return, set to point to an io_iterator_t
				@result kIOReturnSuccess if the iterator could be created
			*/
			IOReturn GetKeySubdirectories(int key, io_iterator_t *outIterator);
			IOReturn GetType(int *outType) ;
			IOReturn GetNumEntries(int *outNumEntries) ;
		
		protected:
			Device&						mUserClient ;
			UserObjectHandle	mKernConfigDirectoryRef ;			
	} ;
	
	class ConfigDirectoryCOM: public ConfigDirectory
	{
		protected:
			typedef ::IOFireWireConfigDirectoryInterface	Interface ;
	
		public:
			ConfigDirectoryCOM(Device& inUserClient) ;
			ConfigDirectoryCOM(Device& inUserClient, UserObjectHandle inDirRef) ;
			virtual ~ConfigDirectoryCOM() ;

		private:
			static Interface sInterface ;

		public:
			// --- IUNKNOWN support ----------------
			static IUnknownVTbl**	Alloc(Device& inUserClient, UserObjectHandle inDirRef) ;
			static IUnknownVTbl**	Alloc(Device& inUserClient) ;
			virtual HRESULT			QueryInterface(REFIID iid, void ** ppv ) ;
		
		protected:
			// --- static methods ------------------
			static IOReturn SUpdate(
									DirRef 	inDir, 
									UInt32 								inOffset) ;
			static IOReturn SGetKeyType(
									DirRef 	inDir, 
									int 								inKey, 
									IOConfigKeyType* 					outType);
			static IOReturn SGetKeyValue_UInt32(
									DirRef 	inDir, 
									int 								inKey, 
									UInt32*								outValue, 
									CFStringRef*						outText);
			static IOReturn SGetKeyValue_Data(
									DirRef 	inDir, 
									int 								inKey, 
									CFDataRef*							outValue, 
									CFStringRef*						outText);
			static IOReturn SGetKeyValue_ConfigDirectory(
									DirRef 	inDir, 
									int 								inKey, 
									DirRef *	outValue, 
									REFIID								iid,
									CFStringRef*						outText);
			static IOReturn SGetKeyOffset_FWAddress(
									DirRef 				inDir, 
									int 				inKey, 
									FWAddress*			outValue, 
									CFStringRef*		text);
			static IOReturn SGetIndexType(
									DirRef 				inDir, 
									int 				inIndex, 
									IOConfigKeyType*	type);
			static IOReturn SGetIndexKey(
									DirRef 				inDir, 
									int 				inIndex, 
									int *				key);
			static IOReturn SGetIndexValue_UInt32(
									DirRef 				inDir, 
									int 				inIndex, 
									UInt32 *			value);
			static IOReturn SGetIndexValue_Data(
									DirRef 				inDir, 
									int 				inIndex, 
									CFDataRef *			value);
			static IOReturn SGetIndexValue_String(
									DirRef 				inDir, 
									int 				inIndex, 
									CFStringRef*		outValue);
			static IOReturn SGetIndexValue_ConfigDirectory(
									DirRef 				inDir, 
									int 				inIndex, 
									DirRef *			outValue,
									REFIID				iid);
			static IOReturn SGetIndexOffset_FWAddress(
									DirRef 				inDir, 
									int 				inIndex, 
									FWAddress*			outValue);
			static IOReturn SGetIndexOffset_UInt32(
									DirRef 				inDir, 
									int 				inIndex, 
									UInt32*				outValue);
			static IOReturn SGetIndexEntry(
									DirRef 				inDir, 
									int 				inIndex, 
									UInt32*				outValue);
			static IOReturn SGetSubdirectories(
									DirRef 				inDir, 
									io_iterator_t*		outIterator);
			static IOReturn SGetKeySubdirectories(
									DirRef 				inDir,
									int 				inKey, 
									io_iterator_t *		outIterator);
			static IOReturn SGetType(
									DirRef 				inDir, 
									int *				outType) ;
			static IOReturn SGetNumEntries(
									DirRef		 		inDir, 
									int *				outNumEntries) ;
	} ;
}
