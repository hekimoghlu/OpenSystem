/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
//
// structure - structural framework for securityd objects
//
#include "structure.h"


//
// NodeCore always has a destructor (because it's virtual),
// but its dump support is conditionally included.
//
NodeCore::~NodeCore()
try {
#if defined(DEBUGDUMP)
	StLock<Mutex> _(mCoreLock);
	mCoreNodes.erase(this);
#endif //DEBUGDUMP
} catch(...) {
    return;
}


//
// Basic object mesh maintainance
//
void NodeCore::parent(NodeCore &p)
{
	StLock<Mutex> _(*this);
	mParent = &p;
}

void NodeCore::referent(NodeCore &r)
{
	StLock<Mutex> _(*this);
	assert(!mReferent);
	mReferent = &r;
}
	
void NodeCore::clearReferent()
{
	StLock<Mutex> _(*this);
	mReferent = NULL;
}


void NodeCore::addReference(NodeCore &p)
{
	StLock<Mutex> _(*this);
	assert(p.mReferent == this);
	mReferences.insert(&p);
}

void NodeCore::removeReference(NodeCore &p)
{
	StLock<Mutex> _(*this);
	mReferences.erase(&p);
}

//
// ClearReferences clears the reference set but does not propagate
// anything; it is NOT recursive.
//
void NodeCore::clearReferences()
{
	StLock<Mutex> _(*this);
	secinfo("ssnode", "%p clearing all %d references", this, int(mReferences.size()));
	mReferences.erase(mReferences.begin(), mReferences.end());
}


//
// Kill should be overloaded by Nodes to implement any cleanup and release
// operations that should happen at LOGICAL death of the represented object.
// This is where you should release ports, close files, etc.
// This default behavior, which you MUST include in your override,
// propagates kills to all active references, recursively.
//
void NodeCore::kill()
{
	StLock<Mutex> _(*this);
	for (ReferenceSet::const_iterator it = mReferences.begin(); it != mReferences.end(); it++)
		(*it)->kill();
	clearReferences();
}


void NodeCore::kill(NodeCore &ref)
{
	StLock<Mutex> _(*this);
	ref.kill();
	removeReference(ref);
}


//
// NodeCore-level support for state dumping.
// Call NodeCore::dumpAll() to debug-dump all nodes.
// Note that enabling DEBUGDUMP serializes all node creation/destruction
// operations, and thus may cause significant shifts in thread interactions.
//
#if defined(DEBUGDUMP)

// The (uncounted) set of all known NodeCores in existence, with protective lock
set<NodeCore *> NodeCore::mCoreNodes;
Mutex NodeCore::mCoreLock;

// add a new NodeCore to the known set
NodeCore::NodeCore()
	: Mutex(Mutex::recursive)
{
	StLock<Mutex> _(mCoreLock);
	mCoreNodes.insert(this);
}

// partial-line common dump text for any NodeCore
// override this to add text to your Node type's state dump output
void NodeCore::dumpNode()
{
 Debug::dump("%s@%p rc=%u", Debug::typeName(*this).c_str(), this, unsigned(refCountForDebuggingOnly()));
	if (mParent)
		Debug::dump(" parent=%p", mParent.get());
	if (mReferent)
		Debug::dump(" referent=%p", mReferent.get());
}

// full-line dump of a NodeCore
// override this to completely re-implement the dump format for your Node type
void NodeCore::dump()
{
 dumpNode();
	if (!mReferences.empty()) {
		Debug::dump(" {");
		for (ReferenceSet::const_iterator it = mReferences.begin(); it != mReferences.end(); it++) {
			Debug::dump(" %p", it->get());
			if ((*it)->mReferent != this)
				Debug::dump("!*INVALID*");
		}
		Debug::dump(" }");
	}
	Debug::dump("\n");
}

// dump all known nodes
void NodeCore::dumpAll()
{
 StLock<Mutex> _(mCoreLock);
	time_t now; time(&now);
	Debug::dump("\nNODE DUMP (%24.24s)\n", ctime(&now));
	for (set<NodeCore *>::const_iterator it = mCoreNodes.begin(); it != mCoreNodes.end(); it++)
		(*it)->dump();
	Debug::dump("END NODE DUMP\n\n");
}

#endif //DEBUGDUMP
