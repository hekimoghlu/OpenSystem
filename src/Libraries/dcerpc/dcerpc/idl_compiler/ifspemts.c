/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
**
**  NAME:
**
**      ifspemts.c
**
**  FACILITY:
**
**      IDL Compiler Backend
**
**  ABSTRACT:
**
**  Routines for the emission of ifspecs, including MIA transfer syntaxes
**
**
*/

#include <nidl.h>
#include <ast.h>
#include <cspell.h>
#include <command.h>
#include <ifspec.h>
#include <bedeck.h>
#include <dutils.h>
#include <mtsbacke.h>

/******************************************************************************/
/*                                                                            */
/*    Return a pointer to the name of the if_spec                             */
/*                                                                            */
/******************************************************************************/
static char *BE_ifspec_name
(
    AST_interface_n_t *ifp,
    BE_output_k_t kind
)
{
    static char retval[100];

    sprintf(retval, "%s_v%ld_%ld_%c_ifspec", BE_get_name(ifp->name),
            (ifp->version%65536), (ifp->version/65536),
            kind == BE_client_stub_k ? 'c' : 's');

    return retval;
}

/******************************************************************************/
/*                                                                            */
/*    Spell the manager epv to the output stream                              */
/*                                                                            */
/******************************************************************************/
void CSPELL_manager_epv
(
    FILE *fid,
    AST_interface_n_t *ifp
)
{
    AST_export_n_t *p_export;
    boolean first_op = true;

    fprintf( fid, "\nstatic %s_v%ld_%ld_epv_t IDL_manager_epv = {\n",
           BE_get_name(ifp->name), (ifp->version%65536), (ifp->version/65536) );

    for( p_export = ifp->exports; p_export != NULL; p_export = p_export->next )
    {
        if (p_export->kind == AST_operation_k)
        {
            if ( ! first_op ) fprintf( fid, "," );
            spell_name( fid, p_export->thing_p.exported_operation->name );
            fprintf( fid, "\n" );
            first_op = false;
        }
    }

    fprintf( fid,"};\n" );
}

/******************************************************************************/
/*                                                                            */
/*    Spell an interface definition and related material to the output stream */
/*                                                                            */
/******************************************************************************/
void CSPELL_interface_def
(
    FILE *fid,
    AST_interface_n_t *ifp,
    BE_output_k_t kind,
    boolean generate_mepv
)
{
    boolean     first;
    long        i, endpoints;
    char const *protseq, *portspec;

    if ((endpoints = ifp->number_of_ports) != 0)
    {
        first = true;
        fprintf(fid,
                "static rpc_endpoint_vector_elt_t IDL_endpoints[%ld] = \n{",
                endpoints);
        for (i = 0; i < endpoints; i++)
        {
            STRTAB_str_to_string(ifp->protocol[i], &protseq);
            STRTAB_str_to_string(ifp->endpoints[i], &portspec);
            if (!first) fprintf(fid, ",\n");
            fprintf(fid,
               "{(unsigned_char_p_t)\"%s\", (unsigned_char_p_t)\"%s\"}",
                   protseq, portspec);
            first = false;
        }
        fprintf(fid, "};\n");
    }

    /* Transfer syntaxes */
    fprintf( fid, "\nstatic rpc_syntax_id_t IDL_transfer_syntaxes[%d] = {\n{\n",
                                                                 1 );
        fprintf(fid,
"{0x8a885d04u, 0x1ceb, 0x11c9, 0x9f, 0xe8, {0x8, 0x0, 0x2b, 0x10, 0x48, 0x60}},");
        fprintf(fid, "\n2}");
    fprintf(fid, "};\n");

    fprintf(fid, "\nstatic rpc_if_rep_t IDL_ifspec = {\n");
    fprintf(fid, "  1, /* ifspec rep version */\n");
    fprintf(fid, "  %d, /* op count */\n", ifp->op_count);
    fprintf(fid, "  %ld, /* if version */\n", ifp->version);
    fprintf(fid, "  {");
    fprintf(fid, "0x%8.8lxu, ", ifp->uuid.time_low);
    fprintf(fid, "0x%4.4x, ", ifp->uuid.time_mid);
    fprintf(fid, "0x%4.4x, ", ifp->uuid.time_hi_and_version);
    fprintf(fid, "0x%2.2x, ", ifp->uuid.clock_seq_hi_and_reserved);
    fprintf(fid, "0x%2.2x, ", ifp->uuid.clock_seq_low);
    fprintf(fid, "{");
    first = true;
    for (i = 0; i < 6; i++)
    {
        if (first)
            first = false;
        else
            fprintf (fid, ", ");
        fprintf (fid, "0x%x", ifp->uuid.node[i]);
    }
    fprintf(fid, "}},\n");
    fprintf(fid, "  2, /* stub/rt if version */\n");
    fprintf(fid, "  {%ld, %s}, /* endpoint vector */\n", endpoints,
                 endpoints ? "IDL_endpoints" : "NULL");
    fprintf(fid, "  {%d, IDL_transfer_syntaxes} /* syntax vector */\n",
                                                                 1 );

    if (kind == BE_server_stub_k)
    {
        fprintf(fid, ",IDL_epva /* server_epv */\n");
        if (generate_mepv)
        {
            fprintf(fid,",(rpc_mgr_epv_t)&IDL_manager_epv /* manager epv */\n");
        }
        else
        {
            fprintf(fid,",NULL /* manager epv */\n");
        }
    }
    else
    {
        fprintf(fid,",NULL /* server epv */\n");
        fprintf(fid,",NULL /* manager epv */\n");
    }
    fprintf(fid, "};\n");
    fprintf(fid,
            "/* global */ rpc_if_handle_t %s = (rpc_if_handle_t)&IDL_ifspec;\n",
            BE_ifspec_name(ifp, kind));
}
