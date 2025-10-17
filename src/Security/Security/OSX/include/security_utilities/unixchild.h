/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
// unixchild - low-level UNIX process child management
//
#ifndef _H_UNIXCHILD
#define _H_UNIXCHILD

#include <security_utilities/utilities.h>
#include <security_utilities/errors.h>
#include <security_utilities/globalizer.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <map>
#include <list>


namespace Security {
namespace UnixPlusPlus {


//
// A Child object represents a (potential) fork-child of your process.
// It could be a clean fork or a fork/exec; this layer doesn't care.
// It is meant to track the UNIX-life of that process.
// Subclass Child or use it as a mix-in.
//
// Child keeps track of all alive children; Child::find<>() can locate them
// by pid if you like. Other children are not collected; you've got to do this
// yourself.
//
class Child {
public:
	Child();
	virtual ~Child();
	
	enum State {
		unborn,				// never forked
		alive,				// last seen alive
		dead,				// coroner confirms death
		stopped,			// stopped due to trace or job control (not implemented)
		abandoned,			// cut loose (via forget())
		invalid				// system says we're all confused about this child
	};
	
	void fork();			// do the forky-forky

	State state() const { return mState; }
	operator bool () const { return mState == alive; }
	pid_t pid() const { assert(mState != unborn); return mPid; }
	
	State check();			// update status on (just) this child and return new state
	void wait();			// wait for (just) this Child to die
	
	void kill(int signal);	// send signal to child (if alive)
	void kill();			// bring me its head, NOW
	
	// status is only available for dead children
	int waitStatus() const;	// raw wait(2) status byte
	bool succeeded() const { return waitStatus() == 0; }
	bool bySignal() const;	// was killed by a signal
	int exitCode() const;	// exit() code; valid only if !bySignal()
	int exitSignal() const;	// signal that killed child; valid only if bySignal()
	bool coreDumped() const; // child dumped core when it died

protected:
	virtual void childAction() = 0; // called in child after fork()
	virtual void parentAction();	// called in parent after fork()
	virtual void dying();	// called when child is confirmed dead
	
	void abandon();			// cut a living child loose (forget about it)
	
private:
	State mState;			// child state
	pid_t mPid;				// pid of child (if born)
	int mStatus;			// exit status (if dead)
	
	bool checkStatus(int options); // check status of this Child (wait4)
	void bury(int status);	// canonical last rites
	static Child *findGeneric(pid_t pid); // find living child by pid
	void tryKill(int signal);
	
	class Bier: public std::list<Child *> {
	public:
		void add(Child *child) { this->push_back(child); }
		void notify();
	};

public:
	// set sharedChildren(true) in library code to leave other children alone
	static void sharedChildren(bool s);
	static bool sharedChildren();
	
	void reset();			// make Child ready to be born again (forgets all state)

	static void checkChildren(); // update status on living offspring

	template <class Subclass>
	static Subclass *find(pid_t pid)
	{ return dynamic_cast<Subclass *>(findGeneric(pid)); }

private:
	struct Children : public Mutex, public std::map<pid_t, Child *> {
		Children() : shared(false) { }
		bool shared;
	};
	static ModuleNexus<Children> mChildren;
};


}	// end namespace UnixPlusPlus
}	// end namespace Security

#endif //_H_UNIXCHILD
