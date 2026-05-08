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
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: './logs/server-error.log',
      out_file: './logs/server-out.log',
      log_file: './logs/combined.log',
      time: true
    },
    {
      name: 'sidya-agent',
      script: 'agent.py',
      cwd: '/root/sidya/agent',
      interpreter: '/root/sidya/.venv/bin/python',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      autorestart: true,
      restart_delay: 3000,
      max_memory_restart: '500M',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      error_file: './logs/agent-error.log',
      out_file: './logs/agent-out.log',
      log_file: './logs/combined.log',
      time: true
    }
  ]
};
