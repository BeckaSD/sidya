module.exports = {
  apps: [
    {
      name: 'sidya-server',
      script: 'server.js',
      cwd: '/root/sidya',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      restart_delay: 3000,
      max_memory_restart: '500M',
      env: { NODE_ENV: 'production', PORT: 3000 },
      error_file: './logs/server-error.log',
      out_file: './logs/server-out.log',
      log_file: './logs/combined.log',
      time: true
    },
    {
      name: 'sidya-agent',
      script: '/root/sidya/.venv/bin/uvicorn',
      args: 'agent:app --host 127.0.0.1 --port 8000 --workers 2',
      cwd: '/root/sidya/agent',
      interpreter: 'none',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      restart_delay: 3000,
      max_memory_restart: '800M',
      env: {
        PYTHONUNBUFFERED: '1',
        PATH: '/root/sidya/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
      },
      error_file: './logs/agent-error.log',
      out_file: './logs/agent-out.log',
      log_file: './logs/combined.log',
      time: true
    }
  ]
};
