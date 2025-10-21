/* spice-connection.vala
 *
 * Copyright (C) Red Hat, Inc
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Felipe Borges <felipeborges@gnome.org>
 *
 */
 
using Spice;

namespace Connections {
    private class SpiceConnection : Connection {
        private Spice.Display display;
        private Spice.Session session;
        private Spice.MainChannel? _main_channel;
        public Spice.MainChannel? main_channel {
            get {
                return _main_channel;
            }

            set {
                _main_channel = value;
                if (_main_channel == null) return;

                main_event_id = main_channel.channel_event.connect (main_event);
            }
        }
        ulong main_event_id;
        ulong channel_new_id;

        public override Gtk.Widget widget { 
            set {
                display = value as Spice.Display;
            }

            get {
                return display as Gtk.Widget;
            }
        }
        
        protected override Gdk.Pixbuf? thumbnail { 
            owned get {
                return display.get_pixbuf();
            }
            
            set {
                return;
            }
        }

        public override bool scaling { 
            set {
                display.scaling = value;
            }

            get {
                if (!connected)
                    return true;

                return display.scaling;
            }
        }

        public string scale_mode { get; set; default = "fit-window"; }
        private bool _enable_audio = true;
        public bool enable_audio {
            set {
                _enable_audio = value;
                session.enable_audio = _enable_audio;
            }

            get {
                return _enable_audio;
            }
        }

        public override int port { get; protected set; default = 3128; }
        
        construct {
            session = new Session();
            Spice.set_session_option (session);
            display = new Spice.Display (session, 0);
            notify["scale-mode"].connect (scale);

            authentication_complete.connect (update_display_authenticated);
        }

        public SpiceConnection (string uuid) {
            this.uuid = uuid;
        }

        public SpiceConnection.from_uri (string uri) {
            this.uuid = Uuid.string_random ();
            this.uri = uri;
        }
        
        public override void send_keys (uint[] keys) {
            display.send_keys (keys, Spice.DisplayKeyEvent.CLICK);
        }

        private void update_display_authenticated () {
            connect_it ();
        }

        public override void connect_it () {
            if (connected) {
                return;
            }
            connected = true;

            main_cleanup ();

            session.host = host;
            session.port = port.to_string ();
            session.password = password;

            if (channel_new_id == 0)
                channel_new_id = session.channel_new.connect (on_channel_new);

            session.connect ();
        }

        public override void disconnect_it () {
            if (connected) {
                var session_object = session as GLib.Object;

                if (channel_new_id > 0) {
                    session_object.disconnect (channel_new_id);
                    channel_new_id = 0;
                }                    
                
                session.disconnect ();
                connected = false;
                main_cleanup ();
            }
        }

        public override void dispose_display () {}

        ~SpiceConnection () {
            debug ("Closing connection with %s", widget.name);

            disconnect_it ();
        }

        private void on_channel_new (Spice.Session session, Spice.Channel channel) {
            if (channel is Spice.MainChannel) {
                main_channel = (channel as Spice.MainChannel);
            }
                
        }

        private void main_event (ChannelEvent event) {
            switch (event) {
                case ChannelEvent.CLOSED:
                    disconnect_it ();
                    break;
                case ChannelEvent.ERROR_AUTH:
                    need_password = true;
                    handle_auth ();
                    disconnect_it ();                    
                    break;
                case ChannelEvent.ERROR_CONNECT:
                case ChannelEvent.ERROR_IO:
                case ChannelEvent.ERROR_LINK:
                case ChannelEvent.ERROR_TLS:
                    debug ("main SPICE channel error: %d", event);
                    disconnect_it ();
                    break;
                case ChannelEvent.OPENED:
                    show ();
                    break;
                default:
                    debug ("unhandled main SPICE channel event: %d", event);
                    break;
            }
        }

        private void main_cleanup () {
            if (main_channel == null) return;

            var o = main_channel as Object;
            o.disconnect(main_event_id);
            main_event_id = 0;
            main_channel = null;
        }

        public void scale () {
            display.scaling = display.expand = false;

            switch (scale_mode) {
                case "resize-desktop":
                    resize_desktop_to_window ();
                    break;
                case "fit-window":
                    scale_to_fit_window ();
                    break;
                case "original":
                    scale_to_original_size ();
                    break;
            }
        }

        private void resize_desktop_to_window () {
            display.resize_guest = true;
        }

        private void scale_to_fit_window () {
            display.scaling = display.hexpand = true;
            display.width_request = display.height_request = 0;
            display.resize_guest = false;
        }

        private void scale_to_original_size () {
            display.width_request = display.get_allocated_width ();
            display.height_request = display.get_allocated_height ();
            display.resize_guest = false;
        }
    }
}