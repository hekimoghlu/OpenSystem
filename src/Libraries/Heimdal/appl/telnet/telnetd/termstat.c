/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
#include "telnetd.h"

RCSID("$Id$");

/*
 * local variables
 */
int def_tspeed = -1, def_rspeed = -1;
#ifdef	TIOCSWINSZ
int def_row = 0, def_col = 0;
#endif

/*
 * flowstat
 *
 * Check for changes to flow control
 */
void
flowstat(void)
{
    if (his_state_is_will(TELOPT_LFLOW)) {
	if (tty_flowmode() != flowmode) {
	    flowmode = tty_flowmode();
	    output_data("%c%c%c%c%c%c",
			IAC, SB, TELOPT_LFLOW,
			flowmode ? LFLOW_ON : LFLOW_OFF,
			IAC, SE);
	}
	if (tty_restartany() != restartany) {
	    restartany = tty_restartany();
	    output_data("%c%c%c%c%c%c",
			IAC, SB, TELOPT_LFLOW,
			restartany ? LFLOW_RESTART_ANY
			: LFLOW_RESTART_XON,
			IAC, SE);
	}
    }
}

/*
 * clientstat
 *
 * Process linemode related requests from the client.
 * Client can request a change to only one of linemode, editmode or slc's
 * at a time, and if using kludge linemode, then only linemode may be
 * affected.
 */
void
clientstat(int code, int parm1, int parm2)
{
    /*
     * Get a copy of terminal characteristics.
     */
    init_termbuf();

    /*
     * Process request from client. code tells what it is.
     */
    switch (code) {
    case TELOPT_NAWS:
#ifdef	TIOCSWINSZ
	{
	    struct winsize ws;

	    def_col = parm1;
	    def_row = parm2;

	    /*
	     * Change window size as requested by client.
	     */

	    ws.ws_col = parm1;
	    ws.ws_row = parm2;
	    ioctl(ourpty, TIOCSWINSZ, (char *)&ws);
	}
#endif	/* TIOCSWINSZ */

    break;

    case TELOPT_TSPEED:
	{
	    def_tspeed = parm1;
	    def_rspeed = parm2;
	    /*
	     * Change terminal speed as requested by client.
	     * We set the receive speed first, so that if we can't
	     * store seperate receive and transmit speeds, the transmit
	     * speed will take precedence.
	     */
	    tty_rspeed(parm2);
	    tty_tspeed(parm1);
	    set_termbuf();

	    break;

	}  /* end of case TELOPT_TSPEED */

    default:
	/* What? */
	break;
    }  /* end of switch */

    netflush();

}
