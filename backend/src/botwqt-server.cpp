/*
botw-qt-backend
Copyright (C) 2023 iTNTPiston

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/
#include <util/base.h>
#include <opencv2/opencv.hpp>
#include "botwqt-metadata.generated.h"
#include "botwqt-server.h"
#include "botwqt-frame.h"

namespace botwqt {

Server::Server() {
	m_server.get_alog().clear_channels(websocketpp::log::alevel::all);
	m_server.get_elog().clear_channels(websocketpp::log::elevel::all);
	m_server.init_asio();

// apparently important
#ifndef _WIN32
	m_server.set_reuse_addr(true);
#endif

	m_server.set_open_handler(std::bind(&Server::on_open, this, std::placeholders::_1));
	m_server.set_close_handler(std::bind(&Server::on_close, this, std::placeholders::_1));
}

void Server::start(uint16_t port) {
	if (m_server.is_listening()) {
		blog(LOG_WARNING, "Server is listening when trying to start");
		return;
	}

	m_server.reset();

	blog(LOG_INFO, "Starting server on port %d", port);
	websocketpp::lib::error_code ec;
	m_server.listen(port, ec);
	if (ec) {
		blog(LOG_ERROR, "Failed to start listening: %s", ec.message().c_str());
		return;
	}

	m_server.start_accept();

	m_server_thread = std::thread(&Server::run_server, this);
	blog(LOG_INFO, "Server started");
}

void Server::stop() {
	if (!m_server.is_listening()) {
		return;
	}
	blog(LOG_INFO, "Stopping server");
	m_server.stop_listening();
	// Close all clients
	{
		std::scoped_lock lock(m_clients_mutex);
		for (auto& client : m_clients) {
			websocketpp::lib::error_code ec;
			m_server.pause_reading(client, ec);
			if (ec) {
				blog(LOG_ERROR, "Failed to pause reading from client: %s", ec.message().c_str());
				continue;
			}
			m_server.close(client, websocketpp::close::status::normal, "Server is shutting down", ec);
			if (ec) {
				blog(LOG_ERROR, "Failed to close client: %s", ec.message().c_str());
			}
		}
	}

	m_broadcast_thread_pool.waitForDone();

	// Wait for all clients to be closed
	while (true) {
		{
			std::scoped_lock lock(m_clients_mutex);
			if (m_clients.empty()) {
				break;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	m_server_thread.join();
	m_server.stop();
	blog(LOG_INFO, "Server stopped");
}

Server::~Server() {
	if(m_server.is_listening()) {
		stop();
	}
}

void Server::send_frame(Frame& frame) {
	// If frame is too bright or too dark, skip
	if (!frame.has_valid_score()) {
        return;
    }
	// If no clients are connected, skip ocr
	// This is not critical so we don't need to lock the mutex
	if (m_clients.empty()) {
		return;
	}

	m_broadcast_thread_pool.start([this, hex(frame.to_hex())](){
		std::scoped_lock lock(m_clients_mutex);
		websocketpp::lib::error_code ec;
		for (auto& client : m_clients) {
			m_server.send(client, hex, websocketpp::frame::opcode::text, ec);
			if (ec) {
				blog(LOG_ERROR, "Failed to send message to client: %s", ec.message().c_str());
			}
		}
	});
}

void Server::run_server() {
    blog(LOG_INFO, "[server] Server thread starting");
	try {
		m_server.run();
	} catch (websocketpp::exception const &e) {
		blog(LOG_ERROR, "[server] websocketpp instance returned an error: %s", e.what());
	} catch (const std::exception &e) {
		blog(LOG_ERROR, "[server] websocketpp instance returned an error: %s", e.what());
	} catch (...) {
		blog(LOG_ERROR, "[server] websocketpp instance returned an unknown error");
	}
	blog(LOG_INFO, "[server] Server thread exiting");
}

void Server::on_open(websocketpp::connection_hdl hdl) {
	blog(LOG_INFO, "[server] New connection");

	websocketpp::lib::error_code ec;
	m_server.send(hdl, "Connected to Backend", websocketpp::frame::opcode::text, ec);
	if (ec) {
		blog(LOG_ERROR, "Failed to send message to client: %s", ec.message().c_str());
	}
	std::scoped_lock lock(m_clients_mutex);
	m_clients.insert(hdl);
}

void Server::on_close(websocketpp::connection_hdl hdl) {
	blog(LOG_INFO, "[server] Connection closed");
	std::scoped_lock lock(m_clients_mutex);
	m_clients.erase(hdl);
}

}