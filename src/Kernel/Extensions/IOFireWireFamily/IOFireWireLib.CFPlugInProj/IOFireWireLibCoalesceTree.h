/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#import <IOKit/IOKitLib.h>

namespace IOFireWireLib {

	// ============================================================
	//
	// CoalesceTree
	//
	// ============================================================
	
	class CoalesceTree
	{
		struct Node
		{
			Node*				left ;
			Node*				right ;
			IOVirtualRange		range ;
		} ;
	
		public:
		
			CoalesceTree() ;
			~CoalesceTree() ;
		
		public:
					
			void	 			CoalesceRange(const IOVirtualRange& inRange) ;
			const UInt32	 	GetCount() const ;
			void			 	GetCoalesceList(IOVirtualRange* outRanges) const ;
	
		protected:
		
			void				DeleteNode(Node* inNode) ;
			void				CoalesceRange(const IOVirtualRange& inRange, Node* inNode) ;
			const UInt32		GetCount(Node* inNode) const ;
			void				GetCoalesceList(IOVirtualRange* outRanges, Node* inNode, UInt32* pIndex) const ;
	
		protected:
		
			Node *	mTop ;
	} ;
	
} // namespace
