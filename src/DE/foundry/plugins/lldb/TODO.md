# LLDB Quirks

 * LLDB doesn't seem to send thread events the way it is supposed to for
   DAP. That means we have nothing to key off of for proper thread commands
   to the peer.

   We'll have to start with a synthetic thread to be able to do that (e.g.
   Thread-1).

   Additionally we'll need to update the thread list after each stop event.

   I'd prefer to keep this quirk to just lldb and not make GDB do the slow
   thing as a result of LLDB.

