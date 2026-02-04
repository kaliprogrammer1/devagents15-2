import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

export async function GET() {
  try {
    const [disk, ps] = await Promise.all([
      execAsync('df -h /').catch(() => ({ stdout: 'Filesystem Size Used Avail Use% Mounted on\n/dev/sda1 0 0 0 0% /' })),
      execAsync('ps -eo pid,ppid,cmd,%cpu,%mem,etime,stat --sort=-%cpu | head -n 15').catch(() => ({ stdout: 'PID PPID CMD %CPU %MEM ELAPSED STAT\n' }))
    ]);

    // System info using 'os' module
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;
    const memPercent = Math.round((usedMem / totalMem) * 100);

    const uptimeSeconds = os.uptime();
    const hours = Math.floor(uptimeSeconds / 3600);
    const minutes = Math.floor((uptimeSeconds % 3600) / 60);
    const seconds = Math.floor(uptimeSeconds % 60);
    const uptimeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

    const loadAvg = os.loadavg()[0]; // 1 min load average
    const cpuCores = os.cpus().length;
    const cpuPercent = Math.min(Math.round((loadAvg / cpuCores) * 100), 100);

    // Parse disk
    const diskLines = disk.stdout.split('\n');
    let diskPercent = 0;
    if (diskLines.length > 1) {
      const diskParts = diskLines[1].split(/\s+/);
      const usePart = diskParts.find(p => p.includes('%'));
      diskPercent = usePart ? parseInt(usePart.replace('%', '')) : 0;
    }

    // Parse processes
    const processLines = ps.stdout.trim().split('\n').slice(1);
    const processes = processLines
      .filter(line => line.trim().length > 0)
      .map((line) => {
        const parts = line.trim().split(/\s+/);
        // parts: [pid, ppid, cmd, %cpu, %mem, etime, stat]
        // cmd might have spaces, so we need to be careful. 
        // We can find the indices if we use fixed columns, but ps -eo is usually space separated.
        const pid = parts[0];
        const cpu = parseFloat(parts[parts.length - 4]);
        const mem = parseFloat(parts[parts.length - 3]);
        const etime = parts[parts.length - 2];
        const stat = parts[parts.length - 1];
        
        // Command is everything between ppid and %cpu
        const cmdStart = line.indexOf(parts[2]);
        const cmdEnd = line.lastIndexOf(parts[parts.length - 4]);
        const fullCmd = line.substring(cmdStart, cmdEnd).trim();
        const name = fullCmd.split(' ').shift()?.split('/').pop() || 'unknown';

        return {
          id: pid,
          name: name,
          status: stat.startsWith('R') || stat.startsWith('S') ? 'running' : 'idle',
          cpu: isNaN(cpu) ? 0 : cpu,
          memory: Math.round((mem / 100) * (totalMem / 1024 / 1024)), // Convert %mem to MB
          runtime: etime,
          lastAction: fullCmd.substring(0, 50)
        };
      });

    // Memory distribution (calculate real values)
    const memDistribution = {
      workspace: 0,
      browser: 0,
      system: 0,
      free: Math.round((freeMem / totalMem) * 100)
    };

    processes.forEach(p => {
      const name = p.name.toLowerCase();
      const mem = p.memory / (totalMem / 1024 / 1024) * 100; // approximation of %
      
      if (name.includes('node') || name.includes('next') || name.includes('bun')) {
        memDistribution.workspace += mem;
      } else if (name.includes('chrom') || name.includes('playwright')) {
        memDistribution.browser += mem;
      } else {
        memDistribution.system += mem;
      }
    });

    // Normalize to 100
    memDistribution.workspace = Math.round(memDistribution.workspace);
    memDistribution.browser = Math.round(memDistribution.browser);
    memDistribution.system = Math.round(memDistribution.system);
    
    // Ensure it sums up correctly with free
    const usedSum = memDistribution.workspace + memDistribution.browser + memDistribution.system;
    const realUsed = 100 - memDistribution.free;
    if (usedSum > 0) {
      const scale = realUsed / usedSum;
      memDistribution.workspace = Math.round(memDistribution.workspace * scale);
      memDistribution.browser = Math.round(memDistribution.browser * scale);
      memDistribution.system = Math.round(memDistribution.system * scale);
    }

    return NextResponse.json({
      success: true,
      stats: {
        cpuUsage: cpuPercent,
        memoryUsage: memPercent,
        diskUsage: diskPercent,
        uptime: uptimeStr,
        activeTasks: processes.filter(p => p.status === 'running').length,
        memDistribution,
        pid: process.pid,
        port: process.env.PORT || 3000
      },
      processes
    });
  } catch (error: any) {
    console.error('Stats error:', error);
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
