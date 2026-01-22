import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'russianDate',
  standalone: true
})
export class RussianDatePipe implements PipeTransform {
  transform(value: string | Date | null | undefined, format: 'date' | 'datetime' | 'time' = 'datetime'): string {
    if (!value) return '';
    
    const date = typeof value === 'string' ? new Date(value) : value;
    if (isNaN(date.getTime())) return '';
    
    // Дата уже в московском времени (хранится как +03:00)
    // Просто форматируем локальное время (JavaScript автоматически учитывает timezone)
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    if (format === 'date') {
      return `${day}.${month}.${year}`;
    } else if (format === 'time') {
      return `${hours}:${minutes}`;
    } else {
      return `${day}.${month}.${year} ${hours}:${minutes}`;
    }
  }
}
